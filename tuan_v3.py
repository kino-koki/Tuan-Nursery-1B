#!/usr/bin/env python3
"""
青小团 v3：定位"训不动"根因 + 词表规模 A/B

假设：13.19M 参数里 tok_emb + head 占 32768×256 = 8.39M（**64%**），
真正做计算的只剩 4.8M。词表缩到 8192 可把网络占比提到 ~87%。
这解释了为什么 val 困惑度卡在 20000+（≈词表大小 = 没学到东西）。

阶段：
    v  词表 A/B（8192 vs 32768，同架构同预算）
    c  用优选词表重训 + 去 AI 味语料 + 生成检验
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys, time
from pathlib import Path

for p in ("/data/workspace/SimplCore_official", "/data/workspace",
          "/data/workspace/SimplCore_official/scripts"):
    sys.path.insert(0, p)

import torch
import torch.nn.functional as F
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))

WS = Path("/data/workspace")
OUT = WS / "TUANV3"; OUT.mkdir(exist_ok=True)
RES = OUT / "result.json"
DIM, LAYERS, BASES, HEADS, SEQ = 256, 8, 32, 8, 256


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def save(k, v):
    d = json.loads(RES.read_text()) if RES.exists() else {}
    d[k] = v
    RES.write_text(json.dumps(d, ensure_ascii=False, indent=2))


AI_PATTERNS = [
    (r"不是(.{1,20}?)而是", "不是…而是…"), (r"——", "长破折号"),
    (r"—", "破折号"), (r"首先[，,].{0,30}其次", "首先/其次"),
    (r"值得一提的?[是,，]", "值得一提的是"), (r"总[而]言之", "总而言之"),
    (r"需要注意的?[是,，]", "需要注意的是"), (r"根据我的知识库", "AI自述"),
    (r"\[\d+\]", "编造引用"),
]


def ai_score(t):
    n = 0
    for pat, _ in AI_PATTERNS:
        n += len(re.findall(pat, t))
    return n


def humanize(t):
    t = re.sub(r"不是(.{1,20}?)而是", r"\1", t)
    t = t.replace("——", "，").replace("—", "，")
    t = t.replace("首先，", "").replace("其次，", "").replace("最后，", "")
    t = t.replace("值得一提的是，", "").replace("总而言之，", "")
    t = t.replace("需要注意的是，", "")
    return t.strip()


def load_lines(n=120000, clean=True):
    src = WS / "big_corpus.txt"
    text = src.read_text(encoding="utf-8", errors="ignore")[:6_000_000]
    seen = set(); out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not (16 <= len(ln) <= 180):
            continue
        h = hash(ln)
        if h in seen:
            continue
        seen.add(h)
        out.append(humanize(ln) if clean else ln)
        if len(out) >= n:
            break
    rnd = random.Random(0); rnd.shuffle(out)
    return out


def make_bpe(lines, vs):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tk = Tokenizer(models.BPE(unk_token="<|unk|>", byte_fallback=True))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=vs, min_frequency=3,
        special_tokens=["<|pad|>", "<|tuan|>", "<|end|>", "<|unk|>"],
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(lines[:40000], trainer=tr)
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


def build(V):
    from simplcore.core import get_preset, SimplCoreModel
    torch.manual_seed(1234)
    c = get_preset("tuan-30m")
    c.dim = DIM; c.num_layers = LAYERS; c.num_bases = BASES
    c.num_heads = HEADS; c.vocab_size = V; c.seq_len = SEQ
    return SimplCoreModel(c)


def train_eval(tk, lines, steps, tag):
    V = tk.get_vocab_size()
    w = pack_wins(tk, lines)
    nval = max(1, w.shape[0] // 20)
    tr, va = w[:-nval], w[-nval:]
    m = build(V)
    N = sum(p.numel() for p in m.parameters())
    emb = m.tok_emb.weight.numel() + sum(p.numel()
                                         for p in m.head.parameters())
    log(f"\n  ── [{tag}] V={V} ──")
    log(f"    总参数 {N/1e6:.2f}M | 词表 {emb/1e6:.2f}M ({emb/N*100:.0f}%)"
        f" | 真正网络 {(N-emb)/1e6:.2f}M")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train(); t0 = time.time()
    for i in range(1, steps + 1):
        x = tr[i % tr.shape[0]: i % tr.shape[0] + 1]
        o = m(x[:, :-1], x[:, 1:])
        l = o[1] if isinstance(o, tuple) else o
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if i % max(1, steps // 3) == 0:
            log(f"      step {i:>5} train {float(l):.4f} [{time.time()-t0:.0f}s]")

    @torch.no_grad()
    def ev(n=25):
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

    lv, acc = ev()
    pv = math.exp(min(lv, 20))
    log(f"    val loss {lv:.4f} | 困惑度 {pv:.2f} | "
        f"next-token {acc*100:.2f}% (随机 {100/V:.3f}%)")
    return dict(V=V, params=N, emb=emb, emb_ratio=round(emb / N, 3),
                loss=round(lv, 4), ppl=round(pv, 2), acc=round(acc, 4),
                steps=steps), m, tk


def stage_v():
    log("=" * 66)
    log("[v] 词表规模 A/B：定位'训不动'的根因")
    log("=" * 66)
    lines = load_lines()
    log(f"  语料行 {len(lines)}（去重 + 去 AI 味）")
    steps = int(os.getenv("V_STEPS", "900"))
    res = {}
    prev = json.loads(RES.read_text()).get("v") if RES.exists() else None
    if prev and str(8192) in prev:
        res[8192] = prev[str(8192)]      # 复用已完成的一半
        log("  复用已完成的 V=8192 结果")
    for vs in (int(x) for x in os.getenv("V_SIZES", "8192,32768").split(",")):
        if vs in res:
            continue
        tk = make_bpe(lines, vs)
        r, m, _ = train_eval(tk, lines, steps, f"V{vs}")
        res[vs] = r
        save("v", res)
        del m
    a, b = res[8192], res[32768]
    gain = (b["ppl"] - a["ppl"]) / max(b["ppl"], 1e-9)
    log("\n" + "=" * 66)
    log(f"  小词表 8192 : ppl {a['ppl']:>9.2f}  acc {a['acc']*100:>6.2f}%  "
        f"词表占比 {a['emb_ratio']*100:.0f}%")
    log(f"  大词表 32768: ppl {b['ppl']:>9.2f}  acc {b['acc']*100:>6.2f}%  "
        f"词表占比 {b['emb_ratio']*100:.0f}%")
    log(f"  困惑度改善 {gain*100:.1f}%  "
        f"{'✅ 超过 1 倍' if gain > 0.5 else '⚠️ 未达 1 倍'}")
    log("=" * 66)
    res["gain"] = round(gain, 4)
    save("v", res)


def stage_c():
    """用优选词表（8192）+ 去 AI 味语料重训，做生成检验。"""
    log("=" * 66)
    log("[c] 优选词表重训 + 去 AI 味 + 生成检验")
    log("=" * 66)
    lines = load_lines()
    tk = make_bpe(lines, 8192)
    steps = int(os.getenv("C_STEPS", "1500"))
    r, m, tk = train_eval(tk, lines, steps, "final")

    @torch.no_grad()
    def gen(prompt, maxlen=70):
        m.eval()
        x = torch.tensor([tk.encode(prompt).ids], dtype=torch.long)
        eid = tk.token_to_id("<|end|>")
        for _ in range(maxlen):
            o = m(x[:, -SEQ:])
            lg = o[0] if isinstance(o, tuple) else o
            nid = int(lg[0, -1].argmax())
            x = torch.cat([x, torch.tensor([[nid]])], dim=1)
            if nid == eid:
                break
        return tk.decode(x[0].tolist())

    prompts = ["今天天气", "我觉得这个", "六安是", "Python 里", "他说"]
    log("\n  生成检验：")
    gens = []
    for p in prompts:
        g = gen(p)
        sc = ai_score(g)
        gens.append(dict(prompt=p, text=g[:90], ai=sc))
        log(f"    {p!r} → {g[:56]!r}  AI味 {sc}")
    uniq = len({g["text"][:28] for g in gens})
    avg = sum(g["ai"] for g in gens) / max(len(gens), 1)
    log(f"\n  不同 prompt 去重 {uniq}/{len(gens)} "
        f"{'✅ 无复读' if uniq == len(gens) else '⚠️ 复读'}")
    log(f"  平均 AI 味分 {avg:.2f}（0 = 无 AI 味）")
    r["gens"] = gens; r["uniq"] = uniq; r["avg_ai"] = round(avg, 2)
    save("c", r)
    torch.save(m.state_dict(), OUT / "tuan_v3.pt")
    log(f"\n[save] {OUT/'tuan_v3.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    {"v": stage_v, "c": stage_c}[a.stage]()
