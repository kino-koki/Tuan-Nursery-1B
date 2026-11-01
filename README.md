# 青小团-童谣 Tuan-Nursery-1B

六安市青奈科技有限责任公司

| 项 | 值 |
|---|---|
| 参数 | **1,001,205,450（1.0012B）** |
| 架构 | SimplCore（SGC 谱全局卷积）+ SimplUnity 融合 |
| 词表 | 自训 ByteLevel BPE，V=32768（非 Qwen） |
| 融合源 | Qwen3-0.6B（Apache-2.0） |

## 融合构成

| | 参数 | 占比 |
|---|---:|---:|
| 真实权重（直搬/切片/重组） | 931,287,040 | **93.0%** |
| SimplCore 自有初始化 | 69,918,410 | 7.0% |

- **逐位直搬**：FFN 的 gate/up/down、两个 layernorm（形状完全对齐）
- **切片**：`q_proj[:1024]`→in_proj、`k_proj`→gate_proj、`o_proj[:,:1024]`→out_proj
- **重组**：teacher 只有 28 层，28 层后用两层真实片段的确定性凸组合
  （α = 0.5 + 0.35·sin(1.7i)）。凸组合**不增加知识**，只给待训练的高质量起点。

## 权重在哪

权重（2.0 GB）在 **Releases** 里，分成 4 个 500MB 卷：

```
cat 青小团1B.part_* > tuan1b.zip
unzip tuan1b.zip
```

> 不用 Git LFS：免费账户 LFS 只有 1GB 存储+1GB/月带宽，2GB 会超额。

## 状态

⚠️ **该权重是待训练的起点，未经过预训练。**
93% 参数来自 Qwen3-0.6B 的几何对齐搬移，能力要靠真机跑数据获得。

## 重要修正（比权重本身更值钱）

实测同架构同预算，只改词表：

| | V=8192 | V=32768 |
|---|---:|---:|
| val 困惑度 | **549.43** | 22000+ |
| next-token 准确率 | **14.52%** | 0.08% |

**建议真机重建时用 8192 词表**——省下的 50M 参数给深度。
`build_2b.py` 已按 8192 写好，跑出来是 1.885B。

## 目录

```
build_1b.py      1B 融合构建
build_2b.py      扩量级到 1.885B（词表 8192）
build_tokenizer.py  自训 BPE
tuan_v2.py       4M 上下文 / 去 AI 味 / 四层闸
tuan_v3.py       词表规模 A/B
bench_real.py    同参数同预算对照 + 架构验证
push_release.py  推 Release（不走 LFS）
setup_torch.sh   环境
```
