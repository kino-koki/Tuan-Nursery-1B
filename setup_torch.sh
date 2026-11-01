#!/bin/bash
# 在沙箱里构建可用的 CPU-only torch（CUDA 后端为符号 stub）
# 用法: source /data/workspace/setup_torch.sh
set -e
ulimit -f unlimited 2>/dev/null || true
MIR=https://mirrors.cloud.tencent.com/pypi/simple
DIST=/usr/local/lib/python3.10/dist-packages

if [ ! -d "$DIST/torch" ]; then
  echo "[1/5] 安装 torch 2.6.0 (no-deps)..."
  pip install --no-cache-dir --no-deps "torch==2.6.0" --index-url $MIR -q
fi
if [ ! -d "$DIST/nvidia/cuda_runtime" ]; then
  echo "[2/5] 安装真 CUDA runtime/cupti/nvrtc/nvtx..."
  pip install --no-cache-dir --no-deps nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvtx-cu12 --index-url $MIR -q
fi
rm -rf "$DIST/nvidia/cu13" 2>/dev/null || true

echo "[3/5] 生成 CUDA 库符号 stub..."
python3 - "$DIST" <<'PYEOF'
import collections, subprocess, os, sys
DIST = sys.argv[1]
LIBDIR = os.path.join(DIST, "torch", "lib")
BASE = os.path.join(DIST, "nvidia")
if subprocess.run(["which","gcc"],capture_output=True).returncode != 0:
    sys.exit("need gcc")
want = {"libcublas.so.12":"cublas","libcublasLt.so.12":"cublas","libcudnn.so.9":"cudnn",
        "libcufft.so.11":"cufft","libcurand.so.10":"curand","libcusparse.so.12":"cusparse",
        "libcusolver.so.11":"cusolver","libnccl.so.2":"nccl","libcusparseLt.so.0":"cusparselt",
        "libnvToolsExt.so.1":"nvtx","libnvrtc.so.12":"cuda_nvrtc"}
g = collections.defaultdict(set)
for f in ["libtorch_cuda.so","libtorch_python.so","libtorch_cpu.so","libc10_cuda.so"]:
    p = os.path.join(LIBDIR,f)
    if not os.path.exists(p): continue
    out = subprocess.run(["nm","-D","--undefined-only",p],capture_output=True,text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2: continue
        s = parts[-1]
        if "@" in s:
            n,v = s.split("@",1)
            if v in want: g[v].add(n)
# 无版本符号：按前缀归库
pref = [("cusparseLt","libcusparseLt.so.0"),("cublasLt","libcublasLt.so.12"),
        ("cublas","libcublas.so.12"),("cudnn","libcudnn.so.9"),("cufft","libcufft.so.11"),
        ("curand","libcurand.so.10"),("cusolver","libcusolver.so.11"),
        ("cusparse","libcusparse.so.12"),("nccl","libnccl.so.2"),("nvtx","libnvToolsExt.so.1")]
p = os.path.join(LIBDIR,"libtorch_cuda.so")
if os.path.exists(p):
    out = subprocess.run(["nm","-D","--undefined-only",p],capture_output=True,text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2: continue
        s = parts[-1]
        if "@" in s or not s.replace("_","").isalnum(): continue
        for pre,lib in pref:
            if s.startswith(pre): g[lib].add(s); break
built = 0
for v in want:            # 遍历全部目标库，空也要建（DT_NEEDED 必须存在）
    g.setdefault(v,set())
for v,names in g.items():
    d = want[v]; outdir = os.path.join(BASE,d,"lib"); os.makedirs(outdir,exist_ok=True)
    c = f"/tmp/_stub_{v}.c"; m = f"/tmp/_stub_{v}.map"
    with open(c,"w") as f:
        f.write("/* stub */\n")
        for n in sorted(names):
            if not n.replace("_","").replace("$","").isalnum(): continue
            f.write(f"long long {n}(void){{return 0;}}\n")
    with open(m,"w") as f: f.write("%s { global: *; };\n"%v)
    r = subprocess.run(["gcc","-shared","-fPIC","-o",os.path.join(outdir,v),c,
                        "-Wl,--version-script="+m],capture_output=True,text=True)
    if r.returncode == 0: built += 1
print(f"  stub 库 {built}/{len(g)}，覆盖 {sum(len(x) for x in g.values())} 个符号")
PYEOF

echo "[4/5] 设置 LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH="$(echo $DIST/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH"

echo "[5/5] 验证..."
python3 -c "import torch;print('  torch',torch.__version__,'| cuda',torch.cuda.is_available())"
echo "OK"
