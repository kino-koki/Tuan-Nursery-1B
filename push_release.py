#!/usr/bin/env python3
"""
把青小团权重推到 GitHub Release（不走 Git LFS）。

为什么不用 LFS：
  * 沙箱装不上 git-lfs（apt 源 403，release 二进制被策略拒绝）
  * 更关键：GitHub 免费账户 LFS 只有 1GB 存储 + 1GB/月带宽，
    我们 2GB 直接超额，后续 clone 也会被限流

改用 Release 附件：
  * 单附件上限 2GB（我们的 zip 2.011GB 略超 → 分卷后每卷 500MB 很安全）
  * 走 api.github.com + uploads.github.com，纯 HTTP，不需要 git-lfs
  * 下载方直接 wget，不需要装 LFS

用法：
    export GH_TOKEN=ghp_xxx
    python3 push_release.py --repo kino-koki/Tuan-Nursery-1B --tag v1.0

token 权限：只需目标仓库的 Contents:write（fine-grained 最安全）。
推送完立即 revoke。
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import urllib.request, urllib.error

WS = Path("/data/workspace")
API = "https://api.github.com"


def req(url, token, method="GET", data=None, headers=None, timeout=60):
    h = {"Authorization": f"Bearer {token}",
         "Accept": "application/vnd.github+json",
         "User-Agent": "tuan-push/1.0"}
    if headers:
        h.update(headers)
    body = json.dumps(data).encode() if data is not None else None
    if body:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def upload_asset(upload_url, token, path: Path, tries=3):
    """上传单个附件。"""
    name = path.name
    size = path.stat().st_size
    url = upload_url.split("{")[0] + f"?name={urllib.parse.quote(name)}"
    for t in range(1, tries + 1):
        try:
            with open(path, "rb") as f:
                data = f.read()
            r = urllib.request.Request(url, data=data, method="POST", headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/octet-stream",
                "User-Agent": "tuan-push/1.0"})
            with urllib.request.urlopen(r, timeout=1800) as resp:
                d = json.loads(resp.read() or b"{}")
                print(f"    ✅ {name}  {size/1e6:.0f} MB  → "
                      f"{d.get('browser_download_url','')[:70]}", flush=True)
                return True
        except Exception as e:
            print(f"    ⚠️ {name} 第 {t} 次失败: {type(e).__name__} "
                  f"{str(e)[:80]}", flush=True)
            if t < tries:
                time.sleep(5 * t)
    return False


def main():
    import urllib.parse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--tag", default="v1.0")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    token = os.getenv("GH_TOKEN")
    if not token:
        print("✗ 需要 GH_TOKEN 环境变量")
        sys.exit(1)

    print(f"[auth] 校验 token 与仓库 {a.repo}")
    st, me = req(f"{API}/user", token)
    if st != 200:
        print(f"✗ token 无效 ({st}): {me.get('error','')[:150]}")
        sys.exit(1)
    print(f"  用户: {me.get('login')}")

    st, repo = req(f"{API}/repos/{a.repo}", token)
    if st != 200:
        print(f"✗ 仓库不可访问 ({st}): {repo.get('error','')[:150]}")
        print("  确认：1) 仓库名对不对  2) token 有没有这个仓库的写权限")
        sys.exit(1)
    print(f"  仓库: {repo.get('full_name')}  私有={repo.get('private')}")
    perm = repo.get("permissions", {})
    if not perm.get("push"):
        print("✗ token 对该仓库无 push 权限")
        sys.exit(1)
    print("  ✅ 有 push 权限")

    # 待传文件
    files = []
    for p in sorted(WS.glob("青小团1B.part_*")):
        files.append(p)
    for n in ("青小团_轻量包_脚本与词表.zip", "关键发现与扩量级方案.md",
              "青小团1B_交付说明.md"):
        p = WS / n
        if p.exists():
            files.append(p)
    total = sum(f.stat().st_size for f in files)
    print(f"\n[files] {len(files)} 个 / {total/1e9:.2f} GB")
    for f in files:
        print(f"  {f.stat().st_size/1e6:>8.0f} MB  {f.name}")

    if a.dry_run:
        print("\n[dry-run] 未实际上传")
        return

    # 创建 release
    print(f"\n[release] 创建 {a.tag}")
    st, rel = req(f"{API}/repos/{a.repo}/releases", token, "POST",
                  dict(tag_name=a.tag, name=f"青小团-童谣 Tuan-Nursery-1B {a.tag}",
                       body=("青小团-童谣（Tuan-Nursery-1B）\n\n"
                             "六安市青奈科技有限责任公司\n\n"
                             "- 参数 1.0012B\n"
                             "- 架构 SimplCore(SGC) + SimplUnity 融合\n"
                             "- 93% 参数来自 Qwen3-0.6B 真实权重直搬/切片/重组\n"
                             "- 自训 BPE 词表 V=32768（非 Qwen）\n\n"
                             "**权重文件被分成 4 卷**，下载后用：\n"
                             "```\ncat 青小团1B.part_* > tuan1b.zip\nunzip tuan1b.zip\n```\n\n"
                             "⚠️ 该权重是**待训练的起点**，未经过预训练。"),
                       draft=False, prerelease=False))
    if st not in (200, 201):
        print(f"✗ 创建失败 ({st}): {rel.get('error','')[:250]}")
        sys.exit(1)
    up = rel.get("upload_url", "")
    print(f"  ✅ {rel.get('html_url')}")

    ok = 0
    for f in files:
        if upload_asset(up, token, f):
            ok += 1
    print(f"\n[done] 上传 {ok}/{len(files)}")
    print(f"  {rel.get('html_url')}")
    if ok < len(files):
        print("  ⚠️ 有文件失败，可重跑本脚本（同 tag 会报错，改 --tag）")


if __name__ == "__main__":
    main()
