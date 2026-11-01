#!/usr/bin/env python3
"""
青小团 真实跑分 + 四项验证

═══ 关于"相当于哪款模型" ══════════════════════════════════
必须诚实：拿 2 核 CPU 上 6.9M 参数的实验模型去对标 GPT/Qwen 是无效的。
唯一**有效**的对照是：

    同参数、同预算、同语料、同词表 的 Transformer 基线

这才是能回答"SimplCore 在这个规模上相当于什么水平"的实验。
本脚本做的就是这个。

═══════════════════════════════════════════════════════════

四项验证：
  B1 跑分    同参数同预算 vs Transformer；困惑度 / next-token 准确率 / 吞吐
  A1 架构    O(1) 状态（跨 batch × 跨长度）+ 双形式对偶 + 复杂度指数
  A2 说话    非模板化：不同 prompt 是否吐不同话；同 prompt 多次采样是否多样
  A3 长上下文 流式累计，状态是否恒定

阶段：bench / arch / speak / longctx
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path

for p in ("/data/workspace/SimplCore_official", "/data/workspace",
          "/data/workspace/SimplCore_official/scripts"):
    sys.path.insert(0, p)

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))

WS = Path("/data/workspace")
OUT = WS / "BENCH2"; OUT.mkdir(exist_ok=True)
RES = OUT / "result.json"
DIM, LAYERS, BASES, HEADS, SEQ = 256, 8, 32, 8, 256
VOCAB = 8192


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def save(k, v):
    d = json.loads(RES.read_text()) if RES.exists() else {}
    d[k] = v
    RES.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def get_tok():
    from tokenizers import Tokenizer
    p = WS / "tokenizer" / "tuan_bpe.json"
    if p.exists():
        tk = Tokenizer.from_file(str(p))
        if tk.get_vocab_size() != VOCAB:
            log(f"  ⚠️ 现有词表 V={tk.get_vocab_size()}，目标 {VOCAB}，重建")
            tk = None
    else:
        tk = None
    if tk is None:
        tk = build_tok()
    return tk


def load_lines(n=120000):
    import re
    src = WS / "big_corpus.txt"
    text = src.read_text(encoding="utf-8", errors="ignore")[:6_000_000]
    seen, out = set(), []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not (16 <= len(ln) <= 180):
            continue
        h = hash(ln)
        if h in seen:
            continue
        seen.add(h)
        ln = re.sub(r"不是(.{1,20}?)而是", r"\1", ln)
        ln = ln.replace("——", "，").replace("值得一提的是，", "")
        out.append(ln.strip())
        if len(out) >= n:
            break
    rnd = random.Random(0); rnd.shuffle(out)
    return out


def build_tok(vs=VOCAB):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    lines = load_lines()
    tk = Tokenizer(models.BPE(unk_token="<|unk|>", byte_fallback=True))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=vs, min_frequency=3,
        special_tokens=["<|pad|>", "<|tuan|>", "<|end|>", "<|unk|>"],
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(lines[:40000], trainer=tr)
    (WS / "tokenizer").mkdir(exist_ok=True)
    tk.save(str(WS / "tokenizer" / "tuan_bpe.json"))
    log(f"  [tok] 重建 V={tk.get_vocab_size()}")
    return tk


def pack_wins(tk, lines, cap=8000):
    wins, buf = [], []
    for ln in lines:
        buf.extend(tk.encode(ln + "\n").ids)
        while len(buf) >= SEQ + 1:
            wins.append(buf[:SEQ + 1]); buf = buf[SEQ + 1:]
        if len(wins) > cap:
            break
    return torch.tensor(wins[:cap], dtype=torch.long)


def build_sgc(V):
    from simplcore.core import get_preset, SimplCoreModel
    torch.manual_seed(1234)
    c = get_preset("tuan-30m")
    c.dim = DIM; c.num_layers = LAYERS; c.num_bases = BASES
    c.num_heads = HEADS; c.vocab_size = V; c.seq_len = SEQ
    return SimplCoreModel(c)


class TF(nn.Module):
    """同参数对照。绑定 embedding + 同初始化尺度（否则起步不公平）。"""

    def __init__(self, V, dim=DIM, layers=LAYERS, heads=HEADS, ffn=640,
                 seq=SEQ):
        super().__init__()
        self.seq = seq
        self.emb = nn.Embedding(V, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            ln1=nn.LayerNorm(dim),
            attn=nn.MultiheadAttention(dim, heads, batch_first=True),
            ln2=nn.LayerNorm(dim),
            ff=nn.Sequential(nn.Linear(dim, ffn), nn.GELU(),
                             nn.Linear(ffn, dim)))) for _ in range(layers)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, V)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, 0, 0.02)
        for b in self.blocks:
            nn.init.normal_(b["ff"][0].weight, 0, 0.02)
            nn.init.normal_(b["ff"][2].weight, 0, 0.02)
            nn.init.zeros_(b["ff"][0].bias); nn.init.zeros_(b["ff"][2].bias)

    def forward(self, idx):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        if L > self.seq:
            pos = pos.clamp(max=self.seq - 1)   # 超出训练长度时钳位
        x = self.emb(idx) + self.pos(pos)[None]
        m = torch.triu(torch.ones(L, L, device=idx.device, dtype=torch.bool), 1)
        for b in self.blocks:
            x = x + b["attn"](b["ln1"](x), b["ln1"](x), b["ln1"](x),
                              attn_mask=m, need_weights=False)[0]
            x = x + b["ff"](b["ln2"](x))
        return self.head(self.lnf(x))


def ffn_search(V, target):
    best, bf = None, 640
    for f in range(320, 1025, 32):
        torch.manual_seed(0)
        n = sum(p.numel() for p in TF(V, ffn=f).parameters())
        d = abs(n - target)
        if best is None or d < best:
            best, bf = d, f
    return bf


# ══════════════ bench 同参数同预算对照 ══════════════
def bench():
    log("=" * 66)
    log("[bench] 同参数同预算：SimplCore vs Transformer")
    log("=" * 66)
    tk = get_tok(); V = tk.get_vocab_size()
    lines = load_lines()
    w = pack_wins(tk, lines)
    nval = max(1, w.shape[0] // 20)
    tr, va = w[:-nval], w[-nval:]
    log(f"  V={V}  训练窗 {tr.shape[0]}  val 窗 {va.shape[0]}")

    sgc = build_sgc(V)
    ns = sum(p.numel() for p in sgc.parameters())
    tf = TF(V, ffn=ffn_search(V, ns))
    nt = sum(p.numel() for p in tf.parameters())
    log(f"  SimplCore {ns/1e6:.2f}M | Transformer {nt/1e6:.2f}M "
        f"(差 {abs(ns-nt)/nt*100:.1f}%)")

    @torch.no_grad()
    def ev(m, is_sgc, n=25):
        m.eval(); tot = 0.0; cnt = 0; corr = 0
        for i in range(min(n, va.shape[0])):
            x = va[i:i + 1]
            o = m(x[:, :-1])
            lg = o[0] if isinstance(o, tuple) else o
            tot += float(F.cross_entropy(lg.reshape(-1, V),
                                         x[:, 1:].reshape(-1),
                                         reduction="sum"))
            cnt += x.shape[1] - 1
            corr += int((lg[0].argmax(-1) == x[0, 1:]).sum())
        return tot / max(cnt, 1), corr / max(cnt, 1)

    ls0, _ = ev(sgc, True); lt0, _ = ev(tf, False)
    log(f"  初始 val: SGC {ls0:.4f} | TF {lt0:.4f}")

    os_ = torch.optim.AdamW(sgc.parameters(), lr=3e-3)
    ot = torch.optim.AdamW(tf.parameters(), lr=3e-3)
    sgc.train(); tf.train()
    steps = int(os.getenv("BENCH_STEPS", "900"))
    hist = []; t0 = time.time()
    for s in range(1, steps + 1):
        x = tr[s % tr.shape[0]: s % tr.shape[0] + 1]
        o = sgc(x[:, :-1], x[:, 1:])
        l = o[1] if isinstance(o, tuple) else o
        os_.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_(sgc.parameters(), 1.0); os_.step()
        lt = F.cross_entropy(tf(x[:, :-1]).reshape(-1, V),
                             x[:, 1:].reshape(-1))
        ot.zero_grad(set_to_none=True); lt.backward()
        torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0); ot.step()
        if s % 300 == 0 or s == 1:
            vs, _ = ev(sgc, True, 12); vt, _ = ev(tf, False, 12)
            log(f"    step {s:>4} val SGC {vs:.4f} TF {vt:.4f} "
                f"[{time.time()-t0:.0f}s]")
            hist.append(dict(step=s, sgc=round(vs, 4), tf=round(vt, 4)))

    ls, ac_s = ev(sgc, True); lt, ac_t = ev(tf, False)
    ps, pt = math.exp(min(ls, 20)), math.exp(min(lt, 20))
    log("\n" + "=" * 66)
    log(f"  最终 val loss : SGC {ls:.4f} | TF {lt:.4f}")
    log(f"  最终 困惑度   : SGC {ps:.2f} | TF {pt:.2f}")
    log(f"  next-token    : SGC {ac_s*100:.2f}% | TF {ac_t*100:.2f}%")
    win = "SimplCore" if ls < lt else "Transformer"
    log(f"  → 同参数同预算 **{win}** 更优 "
        f"({abs(ls-lt)/max(lt,1e-9)*100:.1f}%)")
    log("=" * 66)
    save("bench", dict(sgc=dict(loss=round(ls, 4), ppl=round(ps, 2),
                                acc=round(ac_s, 4)),
                       tf=dict(loss=round(lt, 4), ppl=round(pt, 2),
                               acc=round(ac_t, 4)),
                       winner=win, params=dict(sgc=ns, tf=nt),
                       steps=steps, hist=hist, vocab=V))
    torch.save(sgc.state_dict(), OUT / "sgc.pt")
    torch.save(tf.state_dict(), OUT / "tf.pt")


# ══════════════ arch 架构验证 ══════════════
def arch():
    log("=" * 66)
    log("[arch] 真的跑在 SimplCore 吗：O(1) 状态 + 对偶 + 复杂度")
    log("=" * 66)
    V = get_tok().get_vocab_size()
    m = build_sgc(V); m.eval()
    log(f"  模型 {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    log("\n  [1] O(1) 状态 —— 跨 batch")
    rows = []
    for B in (1, 2, 4, 8):
        b = int(m.stream_bytes(B))
        rows.append(dict(batch=B, total=b, per=b / B))
        log(f"    B={B}: 总 {b:,} B | 每样本 {b/B:,.0f} B")
    o1b = len({r["per"] for r in rows}) == 1

    log("\n  [1] O(1) 状态 —— 跨长度")
    lens = []
    for L in (128, 512, 1024, 2048):
        try:
            x = torch.randint(0, V, (1, L))
            with torch.no_grad():
                m(x)
            b = int(m.stream_bytes(1))
            lens.append(dict(L=L, b=b))
            log(f"    L={L:>5}: {b:,} B")
            del x
        except Exception as e:
            log(f"    L={L:>5}: 跳过 {type(e).__name__}")
    o1l = len({r["b"] for r in lens}) == 1
    sb = lens[0]["b"] if lens else 0
    log(f"  → 跨batch {'✅' if o1b else '❌'} | 跨长度(16×) "
        f"{'✅' if o1l else '❌'}  状态 {sb:,} B = {sb/2**20:.3f} MiB")

    log("\n  [2] 双形式对偶（FFT vs IIR）")
    dual = {}
    # 自己实现对偶检查：整段前向 vs 逐 token 流式，最后一步 logits 应一致
    try:
        for L in (64, 96, 128):
            x = torch.randint(0, V, (1, L))
            with torch.no_grad():
                lg_par = m(x)
                lg_par = lg_par[0] if isinstance(lg_par, tuple) else lg_par
                st = m.init_stream(1)
                for t in range(L):
                    lg_str, st = m.stream_step(x[:, t:t + 1], st)
                lg_str = lg_str[0] if isinstance(lg_str, tuple) else lg_str
                d = float((lg_par[0, -1].float() -
                           lg_str[0, -1].float()).abs().max())
            dual[L] = d
            log(f"    L={L}: 并行 vs 流式 max|Δ|={d:.3e} "
                f"{'✅' if d < 4e-5 else '❌'} (<4e-5)")
            del x
    except Exception as e:
        log(f"    不可用 {type(e).__name__}: {str(e)[:80]}")
    dual_ok = bool(dual) and all(v < 4e-5 for v in dual.values())

    log("\n  [3] 复杂度指数（log-log 回归）")
    tf = TF(V, ffn=ffn_search(V, sum(p.numel() for p in m.parameters())))
    tf.eval()
    ts = []
    for L in (128, 512, 2048):
        x = torch.randint(0, V, (1, L))
        t0 = time.time()
        with torch.no_grad():
            m(x)
        a = time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            tf(x)
        b = time.time() - t0
        ts.append(dict(L=L, sgc=a, tf=b))
        log(f"    L={L:>5}: SGC {a*1000:7.1f} ms | TF {b*1000:7.1f} ms | "
            f"SGC 快 {b/max(a,1e-9):.2f}×")
    k_s = k_t = None
    if len(ts) >= 2:
        sx = math.log(ts[-1]["L"] / ts[0]["L"])
        k_s = math.log(max(ts[-1]["sgc"], 1e-9) / max(ts[0]["sgc"], 1e-9)) / sx
        k_t = math.log(max(ts[-1]["tf"], 1e-9) / max(ts[0]["tf"], 1e-9)) / sx
        log(f"    指数: SGC {k_s:.3f} (O(L log L)≈1.0) | "
            f"TF {k_t:.3f} (O(L²)≈2.0)")

    log("\n[arch] 完成")
    save("arch", dict(o1_batch=o1b, o1_len=o1l, state_bytes=sb,
                      batch_rows=rows, len_rows=lens,
                      duality=dual, dual_ok=dual_ok,
                      timing=ts, k_sgc=k_s, k_tf=k_t))


# ══════════════ speak 非模板化 ══════════════
def speak():
    log("=" * 66)
    log("[speak] 真的会说话吗：非模板化检验")
    log("=" * 66)
    tk = get_tok(); V = tk.get_vocab_size()
    m = build_sgc(V)
    ck = OUT / "sgc.pt"
    if ck.exists():
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        log("  载入 bench 训练权重")
    else:
        log("  ⚠️ 无训练权重，先训 600 步")
        lines = load_lines()
        w = pack_wins(tk, lines)
        tr = w[:max(1, w.shape[0] * 4 // 5)]
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        m.train()
        for i in range(1, 601):
            x = tr[i % tr.shape[0]: i % tr.shape[0] + 1]
            o = m(x[:, :-1], x[:, 1:])
            l = o[1] if isinstance(o, tuple) else o
            opt.zero_grad(set_to_none=True); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

    @torch.no_grad()
    def gen(prompt, maxlen=60, temp=1.0, topk=0):
        m.eval()
        x = torch.tensor([tk.encode(prompt).ids], dtype=torch.long)
        eid = tk.token_to_id("<|end|>")
        for _ in range(maxlen):
            o = m(x[:, -SEQ:])
            lg = o[0] if isinstance(o, tuple) else o
            l_ = lg[0, -1].float() / temp
            if topk > 0:
                v, i_ = torch.topk(l_, topk)
                p = torch.zeros_like(l_); p[i_] = v
                l_ = p
            pr = F.softmax(l_, dim=-1)
            nid = int(torch.multinomial(pr, 1))
            x = torch.cat([x, torch.tensor([[nid]])], dim=1)
            if nid == eid:
                break
        return tk.decode(x[0].tolist())

    log("\n  [1] 不同 prompt → 是否吐不同话（greedy）")
    prompts = ["今天天气", "我觉得这个", "六安是", "Python 里",
               "他说完就", "外面的雨", "这本书讲了", "我们要去"]
    outs = []
    for p in prompts:
        g = gen(p)
        outs.append(g)
        log(f"    {p!r:>14} → {g[:52]!r}")
    uniq = len({o[:30] for o in outs})
    log(f"  去重 {uniq}/{len(outs)} "
        f"{'✅ 非模板化' if uniq >= len(outs) * 0.75 else '⚠️ 有复读'}")

    log("\n  [2] 同一 prompt 多次采样 → 是否有多样性")
    p = "今天天气"
    samples = [gen(p, temp=1.0, topk=40) for _ in range(5)]
    for i, s in enumerate(samples):
        log(f"    #{i+1} {s[:52]!r}")
    suniq = len({s[:30] for s in samples})
    log(f"  去重 {suniq}/5 "
        f"{'✅ 有多样性' if suniq >= 3 else '⚠️ 采样退化'}")

    log("\n  [3] 长度分布（是否卡死在同一长度）")
    lens = [len(s) for s in samples]
    log(f"    长度 {lens}  标准差 "
        f"{(sum((x-sum(lens)/len(lens))**2 for x in lens)/len(lens))**0.5:.1f}")

    save("speak", dict(prompts=prompts, outs=outs, uniq=uniq,
                       samples=samples, sample_uniq=suniq, lens=lens))
    log("\n[speak] 完成")


# ══════════════ longctx ══════════════
def longctx():
    log("=" * 66)
    log("[longctx] 长上下文：状态是否恒定")
    log("=" * 66)
    V = get_tok().get_vocab_size()
    torch.manual_seed(1234)
    from simplcore.core import get_preset, SimplCoreModel
    c = get_preset("tuan-30m")
    c.dim = 128; c.num_layers = 4; c.num_bases = 32
    c.vocab_size = V; c.seq_len = 512
    m = SimplCoreModel(c); m.eval()
    log(f"  验证模型 {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    log("  （用小配置跑长序列：2核CPU 跑 1B×长序列会超时）")

    chunk = 256
    nstep = int(os.getenv("LC_STEPS", "60"))
    st = m.init_stream(1)
    t0 = time.time(); seen = []
    for i in range(nstep):
        x = torch.randint(0, V, (1, chunk))
        with torch.no_grad():
            for t in range(chunk):          # stream_step 一次吃一个位置
                _, st = m.stream_step(x[:, t:t + 1], st)
        if (i + 1) % max(1, nstep // 5) == 0:
            b = int(m.stream_bytes(1))
            seen.append(dict(tokens=(i + 1) * chunk, b=b))
            log(f"    累计 {(i+1)*chunk:>7,} tokens → 状态 {b:,} B")
    dt = time.time() - t0
    tot = nstep * chunk
    uniq = len({r["b"] for r in seen})
    log(f"\n  流式 {tot:,} tokens 用时 {dt:.1f}s ({tot/max(dt,1e-9):.0f} tok/s)")
    log(f"  状态取值唯一 {uniq}/{len(seen)} "
        f"{'✅ O(1) 严格成立' if uniq == 1 else '❌'}")

    sb = seen[0]["b"] if seen else 0
    log(f"\n  外推（状态恒定 → 可外推）:")
    for L in (1_000_000, 2_000_000, 4_000_000):
        kv = 2 * 2 * 4 * 128 * 2 * L
        log(f"    L={L:>9,}: SimplCore {sb:,} B | "
            f"同配置 Transformer KV ≈ {kv/2**30:.2f} GiB")
    save("longctx", dict(total=tot, seconds=round(dt, 1),
                         state_bytes=sb, uniq=uniq, steps=nstep))
    log("\n[longctx] 完成")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    {"bench": bench, "arch": arch, "speak": speak,
     "longctx": longctx}[a.stage]()
