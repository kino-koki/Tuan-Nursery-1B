#!/usr/bin/env python3
"""
青小团-童谣 Tuan-1B：真正 1.001B 参数，SimplUnity 融合真实开源权重。

参数预算（自算，可复现）
------------------------
    dim D=1024, ffn F=3072, bases=32, layers L=74, vocab V=32768（自训 BPE）

    每层:
      mixer  3·D² (in/gate/out proj) + D·4 (local_conv) + bases·D + D
             + bases·2 + 1 + D(norm)          = 3,184,705
      ffn    3·D·F + D(norm)                  = 9,438,208
                                    小计       12,622,913
    74 层                                     =   934,095,562
    emb + head (untied) 2·V·D                 =    67,108,864
    ln_f                                      =         1,024
    ─────────────────────────────────────────────────────────
    总计                                       = 1,001,205,450  ≈ 1.0012 B

SimplUnity 融合（真实开源权重来源：Qwen3-0.6B，Apache-2.0）
----------------------------------------------------------
形状**完全对齐**的部分逐位直搬（真实预训练权重）：
    input_layernorm      [1024]        → mixer.norm.weight      [1024]
    post_attention_ln    [1024]        → ffn.norm.weight        [1024]
    mlp.gate_proj        [3072,1024]   → ffn.gate.weight
    mlp.up_proj          [3072,1024]   → ffn.up.weight
    mlp.down_proj        [1024,3072]   → ffn.down.weight
形状**可切片**的部分（fragment，真实权重切片）：
    q_proj [2048,1024] 前半 → mixer.in_proj   [1024,1024]
    k_proj [1024,1024]      → mixer.gate_proj [1024,1024]
    o_proj [1024,2048] 前半 → mixer.out_proj  [1024,1024]
形状**无对应**的部分（SimplCore 独有，用自己的初始化）：
    raw_rho / raw_theta / coef / feed / out_scale / local_conv
    —— 振荡器参数在 Transformer 里根本没有对应物，只能按 sgc 约定初始化。

teacher 只有 28 层，student 有 74 层 → 第 28 层以后用**结构化片段重组**：
从真实 28 层中取两层做凸组合（α 由层号决定）生成新片段，避免简单复制。
★ 必须诚实：重组不增加知识，只提供**待训练的高质量起点**和容量。

内存策略
--------
1B bf16 = 2.0 GB > 沙箱 2 GB RAM，无法整体驻留。
用 safe_open 懒读 teacher（只取当前张量），student 分 shard 构建：
每 shard ≈ 15 层（189M 参数 / 378 MB），填满即 save_file 落盘并释放。
"""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

WS = Path("/data/workspace")
OUT = WS / "TUAN1B"
OUT.mkdir(exist_ok=True)
SRC = WS / "mig" / "model.safetensors"
LOG = OUT / "build.log"

# ── 架构 ──
D, F, BASES, L, V = 1024, 3072, 32, 74, 32768
SPECIAL_N = 8


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)

    t0 = time.time()
    device = "cpu"

    # 载入自定义词表，确认 V
    tok_dir = WS / "tokenizer"
    if (tok_dir / "tuan_bpe.json").exists():
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(tok_dir / "tuan_bpe.json"))
        V = tk.get_vocab_size()
        log(f"[cfg] 自定义词表 V={V}（自训 BPE，非 Qwen）")
    else:
        log("[cfg] ✗ 未找到自定义词表，使用默认 32768")

    # 预算核对
    per_layer = (3 * D * D + 4 * D + BASES * D + D + BASES * 2 + 1 + D) \
        + (3 * D * F + D)
    total = per_layer * L + 2 * V * D + D
    log(f"[cfg] D={D} F={F} bases={BASES} L={L} V={V}")
    log(f"[cfg] 每层 {per_layer:,}  总 {total:,} = {total/1e9:.4f} B")

    sf = safe_open(str(SRC), framework="pt", device="cpu")
    tkeys = set(sf.keys())
    log(f"[src] teacher 张量 {len(tkeys)} 个")

    def get(k):
        return sf.get_tensor(k) if k in tkeys else None

    # teacher 层配置
    TD = 1024
    t_layers = max([int(k.split(".")[2]) for k in tkeys
                    if k.startswith("model.layers.")] or [0]) + 1
    log(f"[src] teacher 层数 {t_layers}")

    # ── SimplUnity：真实片段提取 ──
    def real_block(j):
        """取 teacher 第 j 层的真实片段。"""
        p = f"model.layers.{j}."
        return dict(
            mixer_norm=get(p + "input_layernorm.weight"),
            ffn_norm=get(p + "post_attention_layernorm.weight"),
            gate=get(p + "mlp.gate_proj.weight"),
            up=get(p + "mlp.up_proj.weight"),
            down=get(p + "mlp.down_proj.weight"),
            q=get(p + "self_attn.q_proj.weight"),
            k=get(p + "self_attn.k_proj.weight"),
            v=get(p + "self_attn.v_proj.weight"),
            o=get(p + "self_attn.o_proj.weight"),
        )

    def fragments(i):
        """第 i 层用哪些真实片段。

        i < t_layers     → 直接对应层（逐位直搬）
        i >= t_layers    → 两层真实片段的凸组合（α 由层号决定）
        """
        if i < t_layers:
            return real_block(i), 1.0, None
        # 结构化重组：取两个相距较远的真实层做凸组合
        j1 = i % t_layers
        j2 = (i * 7 + 3) % t_layers
        if j1 == j2:
            j2 = (j2 + 1) % t_layers
        alpha = 0.5 + 0.35 * math.sin(i * 1.7)   # 确定性，可复现
        return real_block(j1), alpha, real_block(j2)

    def mixer_from(blk, alpha, blk2):
        """把 Transformer 注意力权切片成 SimplCore mixer 片段。"""
        def mix(a, b):
            if a is None:
                return None
            if b is None or alpha >= 1.0:
                return a.to(torch.float32)
            return (alpha * a.to(torch.float32)
                    + (1 - alpha) * b.to(torch.float32))
        q = mix(blk["q"], blk2["q"] if blk2 else None)
        k = mix(blk["k"], blk2["k"] if blk2 else None)
        v = mix(blk["v"], blk2["v"] if blk2 else None)
        o = mix(blk["o"], blk2["o"] if blk2 else None)
        out = {}
        # in_proj ← q_proj 前半（q 有 2× 头数，取前半即 1024 行）
        if q is not None and q.shape[0] >= D:
            out["in_proj"] = q[:D, :D].contiguous()
        if k is not None and k.shape == (D, D):
            out["gate_proj"] = k.contiguous()
        elif v is not None and v.shape == (D, D):
            out["gate_proj"] = v.contiguous()
        if o is not None and o.shape[1] >= D:
            out["out_proj"] = o[:, :D].contiguous()
        return out

    # ── 振荡器参数（SimplCore 独有，按 sgc 约定初始化）──
    sys.path.insert(0, str(WS / "SimplCore_official"))
    sys.path.insert(0, str(WS))
    try:
        import simplcore.core.sgc as sgc
        rho0, theta0 = sgc._init_oscillator_grid(BASES, 8192.0, "logfreq")
        raw_rho = sgc._logit(rho0)
        raw_theta = sgc._logit(theta0 / math.pi)
        log("[init] 用 sgc._init_oscillator_grid 初始化振荡器")
    except Exception as e:
        log(f"[init] sgc 不可用（{type(e).__name__}），用对数网格回退")
        lam = torch.exp(torch.linspace(0, math.log(8192.0), BASES))
        rho0 = torch.exp(-1.0 / lam)
        theta0 = torch.exp(torch.linspace(
            math.log(math.pi / 8192.0), math.log(math.pi / 2 * 0.9), BASES))
        eps = 1e-4
        raw_rho = torch.log(rho0.clamp(eps, 1 - eps)) - \
            torch.log1p(-rho0.clamp(eps, 1 - eps))
        raw_theta = torch.log(theta0.clamp(eps, 1 - eps)) - \
            torch.log1p(-theta0.clamp(eps, 1 - eps))

    # ── 分片构建 ──
    shard: dict[str, torch.Tensor] = {}
    shard_n = 0
    shard_idx = 0
    n_real = 0
    n_init = 0
    SHARD_PARAMS = 200_000_000     # 每片 ≤ 200M 参数（400 MB bf16）

    def flush():
        nonlocal shard, shard_n, shard_idx
        if not shard:
            return
        path = OUT / f"model-{shard_idx:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in shard.items()},
                  str(path), metadata={"format": "pt"})
        mb = path.stat().st_size / 1e6
        log(f"  [shard {shard_idx}] {len(shard)} 张量 / {shard_n/1e6:.1f}M 参数 "
            f"/ {mb:.0f} MB")
        shard = {}
        shard_n = 0
        shard_idx += 1

    def put(name, t):
        nonlocal shard_n
        t = t.to(torch.bfloat16).contiguous()
        if not torch.isfinite(t.float()).all():
            raise RuntimeError(f"非有限张量 {name}")
        shard[name] = t
        shard_n += t.numel()
        if shard_n >= SHARD_PARAMS:
            flush()

    log("\n[build] 开始分片构建")

    # 1) embedding / head（词表换了 → 无法从 teacher 直搬，自定义初始化）
    log("  [emb] 自训 BPE 词表，Teacher 词表 151936 无法对齐 → 标准初始化")
    emb = torch.empty(V, D, dtype=torch.float32)
    torch.nn.init.normal_(emb, 0, 0.02)
    put("tok_emb.weight", emb); del emb
    head = torch.empty(V, D, dtype=torch.float32)
    torch.nn.init.normal_(head, 0, 0.02)
    put("head.weight", head); del head
    n_init += 2 * V * D

    # 2) 逐层
    for i in range(L):
        blk, alpha, blk2 = fragments(i)
        real = i < t_layers

        # ---- mixer ----
        mx = mixer_from(blk, alpha, blk2)
        for nm, shape in (("in_proj", (D, D)), ("gate_proj", (D, D)),
                          ("out_proj", (D, D))):
            t = mx.get(nm)
            if t is not None and tuple(t.shape) == shape:
                put(f"blocks.{i}.mixer.{nm}.weight", t)
                n_real += t.numel()
            else:
                w = torch.empty(shape, dtype=torch.float32)
                torch.nn.init.normal_(w, 0, 0.02)
                put(f"blocks.{i}.mixer.{nm}.weight", w)
                n_init += w.numel()
                del w

        # 振荡器（SimplCore 独有）
        put(f"blocks.{i}.mixer.raw_rho", raw_rho.clone())
        put(f"blocks.{i}.mixer.raw_theta", raw_theta.clone())
        coef = torch.randn(BASES, D) * 0.02
        put(f"blocks.{i}.mixer.coef", coef); del coef
        put(f"blocks.{i}.mixer.feed", torch.ones(D))
        put(f"blocks.{i}.mixer.out_scale", torch.ones(1))
        put(f"blocks.{i}.mixer.local_conv.weight",
            torch.randn(D, 1, 4) * 0.02)
        n_init += BASES * D + D + BASES * 2 + 1 + 4 * D

        # norm：真实直搬
        for our_nm, t_nm in (("mixer.norm", "mixer_norm"),
                             ("ffn.norm", "ffn_norm")):
            t = blk[t_nm]
            if t is not None and tuple(t.shape) == (D,):
                if blk2 is not None and alpha < 1.0 and blk2[t_nm] is not None:
                    t = alpha * t.float() + (1 - alpha) * blk2[t_nm].float()
                put(f"blocks.{i}.{our_nm}.weight", t)
                n_real += D
            else:
                w = torch.ones(D)
                put(f"blocks.{i}.{our_nm}.weight", w)
                n_init += D

        # ---- FFN：真实直搬（形状完全对齐）----
        for our_nm, t_nm, shape in (("gate", "gate", (F, D)),
                                    ("up", "up", (F, D)),
                                    ("down", "down", (D, F))):
            t = blk[t_nm]
            ok = t is not None and tuple(t.shape) == shape
            if ok:
                if blk2 is not None and alpha < 1.0 and blk2[t_nm] is not None \
                        and tuple(blk2[t_nm].shape) == shape:
                    t = alpha * t.float() + (1 - alpha) * blk2[t_nm].float()
                put(f"blocks.{i}.ffn.{our_nm}.weight", t)
                n_real += t.numel()
            else:
                w = torch.empty(shape, dtype=torch.float32)
                torch.nn.init.normal_(w, 0, 0.02)
                put(f"blocks.{i}.ffn.{our_nm}.weight", w)
                n_init += w.numel()
                del w

        if i % 10 == 0 or i == L - 1:
            log(f"    layer {i:>3}/{L}  {'真实直搬' if real else '片段重组'} "
                f"(α={alpha:.3f})  [{time.time()-t0:.0f}s]")

    put("ln_f.weight", torch.ones(D))
    n_init += D
    flush()

    total_built = n_real + n_init
    log(f"\n[done] 参数 {total_built:,} = {total_built/1e9:.4f} B")
    log(f"  来自真实开源权重（直搬/切片/重组）: {n_real:,} "
        f"({n_real/total_built*100:.1f}%)")
    log(f"  SimplCore 自有初始化            : {n_init:,} "
        f"({n_init/total_built*100:.1f}%)")
    log(f"  shard 数 {shard_idx}  用时 {time.time()-t0:.0f}s")

    meta = dict(
        model="青小团-童谣 Tuan-Nursery-1B",
        vendor="六安市青奈科技有限责任公司",
        arch="SimplCore(SGC) + SimplUnity 融合",
        params=total_built, params_B=round(total_built / 1e9, 4),
        config=dict(dim=D, ffn=F, num_bases=BASES, num_layers=L,
                    vocab_size=V, tokenizer="自训 ByteLevel BPE（非 Qwen）"),
        fusion=dict(real_params=n_real, init_params=n_init,
                    real_ratio=round(n_real / total_built, 4),
                    source="Qwen3-0.6B (Apache-2.0)",
                    direct=["ffn.gate/up/down", "mixer.norm", "ffn.norm"],
                    sliced=["mixer.in_proj ← q_proj[:D]",
                            "mixer.gate_proj ← k_proj",
                            "mixer.out_proj ← o_proj[:, :D]"],
                    recomposed=f"layer {t_layers}..{L-1} 两层凸组合(α 确定性)",
                    own=["raw_rho", "raw_theta", "coef", "feed",
                         "out_scale", "local_conv", "tok_emb", "head"]),
        shards=shard_idx,
    )
    (OUT / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"[save] {OUT/'config.json'}")


if __name__ == "__main__":
    main()
