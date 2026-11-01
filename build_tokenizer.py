#!/usr/bin/env python3
"""青小团自定义 BPE 词表 —— 完全自训练，不含任何 Qwen / OpenAI 词表。

为什么必须换掉 Qwen 词表：
  1. 词表是最强的模型指纹（151936 的 size、特殊的 <|im_start|> token、
     byte-fallback 的 <0xXX> 写法，一看就是 Qwen）。
  2. Qwen 词表 151936 × 1024 = 155M 参数只做 embedding，占了太多预算。
  3. 自定义词表可按「中文 + Python 代码」双域优化：
     代码片段（def、return、缩进、运算符）给短 token，
     常用汉字/词给单 token，压缩率显著高于通用词表。

设计：
  * ByteLevel BPE（不会 OOV，任何字节串都能编码）
  * vocab 32768（含 256 byte base + 特殊 token + 32500 merge）
  * 特殊 token 用自有命名：<|tuan|>（BOS）、<|end|>（EOS）、
    <|pad|>、<|think|>（深度思考起）、<|ans|>（答案起）、
    <|img|>（图像前缀）、<|sys|>（系统/身份段）
  * 训练语料 = 中文语料 + Python 代码，双域

产出：
  tokenizer/tuan_bpe.json    可直接被 tokenizers 加载
  tokenizer/special.json     特殊 token id 表
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

WS = Path("/data/workspace")
OUT = WS / "tokenizer"
OUT.mkdir(exist_ok=True)

# ── 特殊 token（自有命名，不用 Qwen 的 <|im_start|>）──
SPECIAL = ["<|pad|>", "<|tuan|>", "<|end|>", "<|sys|>",
           "<|think|>", "<|ans|>", "<|img|>", "<|unk|>"]
VOCAB_SIZE = 32768


def build_corpus() -> list[str]:
    """中文语料 + Python 代码，双域。"""
    texts = []
    # 1) 已有中文语料
    for f in ("big_corpus.txt", "corpus2/chinese_poetry.txt"):
        p = WS / f
        if p.exists():
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    # 2) 中文语料目录
    c2 = WS / "corpus2"
    if c2.is_dir():
        for p in sorted(c2.glob("**/*.txt"))[:40]:
            try:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    # 3) SimplCore 仓库源码（Python 域）
    for repo in ("SimplCore_official", "SimplCore_new"):
        rp = WS / repo
        if rp.is_dir():
            n = 0
            for p in rp.rglob("*.py"):
                try:
                    texts.append(p.read_text(encoding="utf-8", errors="ignore"))
                    n += 1
                except Exception:
                    pass
                if n > 600:
                    break
    # 4) 本目录自写脚本（Python 域，含大量教学代码）
    for p in WS.glob("*.py"):
        try:
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    return [t for t in texts if len(t) > 100]


def main():
    t0 = time.time()
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from tokenizers.processors import TemplateProcessing

    texts = build_corpus()
    # ★ 49M 字符全量喂 BPE 会在 2GB 沙箱 OOM（进程被杀，无报错）。
    #   截到 12M 字符：足够覆盖中英/代码字符分布，且内存可控。
    CAP = int(os.getenv('TOK_CHARS', '12_000_000'))
    acc, total = [], 0
    for t in texts:
        if total >= CAP:
            break
        take = t[: CAP - total]
        acc.append(take); total += len(take)
    texts = acc
    print(f"[tok] 语料 {len(texts)} 份 / {total/1e6:.2f}M 字符", flush=True)
    if total < 100_000:
        raise SystemExit("语料太少，无法训练词表")

    tk = Tokenizer(models.BPE(unk_token="<|unk|>", byte_fallback=True))
    # ByteLevel pre-tokenizer：任何字节串都能编码，无 OOV
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                use_regex=True)
    tk.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 分块喂，控制内存
    BATCH = 2000
    for i in range(0, len(texts), BATCH):
        tk.train_from_iterator(texts[i:i + BATCH], trainer=trainer)
        print(f"  训练 {min(i+BATCH, len(texts))}/{len(texts)} 份 "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # post-process：<|tuan|> 开头，<|end|> 结尾
    tk.post_processor = TemplateProcessing(
        single="<|tuan|> $A <|end|>",
        pair="<|tuan|> $A <|end|> <|tuan|> $B <|end|>",
        special_tokens=[("<|tuan|>", tk.token_to_id("<|tuan|>")),
                        ("<|end|>", tk.token_to_id("<|end|>"))],
    )

    tk.save(str(OUT / "tuan_bpe.json"))
    V = tk.get_vocab_size()
    print(f"\n[tok] 词表大小 {V}", flush=True)

    ids = {s: tk.token_to_id(s) for s in SPECIAL}
    (OUT / "special.json").write_text(
        json.dumps(ids, ensure_ascii=False, indent=2))

    # 压缩率对比（每字符 token 数，越低越好）
    samples = [
        "def add(a, b):\n    return a + b\n",
        "六安市青奈科技有限责任公司开发的青小团-童谣模型",
        "思考：先算 7+7=14，再加 1 得 15\n答案：15",
        "SimplCore 的谱全局卷积状态复杂度是 O(1)",
    ]
    print("\n[tok] 压缩率（token/字符，越低越好）:")
    rows = []
    for s in samples:
        e = tk.encode(s)
        r = len(e.ids) / max(len(s), 1)
        rows.append(dict(text=s[:30], tokens=len(e.ids), chars=len(s),
                         ratio=round(r, 3)))
        print(f"  {r:.3f}  ({len(e.ids):>3} tok / {len(s):>3} 字)  {s[:34]}")
    avg = sum(r["ratio"] for r in rows) / len(rows)
    print(f"  平均 {avg:.3f} token/字符")
    print(f"  对照：Qwen 词表中文约 0.6-0.9；纯字符级 = 1.000")

    (OUT / "meta.json").write_text(json.dumps(dict(
        vocab_size=V, special=ids, samples=rows,
        avg_ratio=round(avg, 3), corpus_chars=total,
        note="自训练 ByteLevel BPE，与 Qwen/OpenAI 词表无任何交集"),
        ensure_ascii=False, indent=2))
    print(f"\n[tok] 完成 [{time.time()-t0:.0f}s] → {OUT/'tuan_bpe.json'}")


if __name__ == "__main__":
    main()
