# MyLeetGpu

MyLeetGpu 是一个面向单机可信操作者的 GPU 编程、正确性验证与性能比较环境。它把 CUDA C++、Triton (Python) Kernel 和高层 PyTorch (Python) 作为三种一等实现类型；当前内置八道 CUDA/Triton Kernel 题，以及多头自注意力（MHA）和分组查询自注意力（GQA）两道 PyTorch 题。平台提供 Monaco 编辑器、异步隔离 Judge、手动性能版本、持久化 benchmark 和同语言统一口径比较。

> 安全边界：默认只发布到 `127.0.0.1`。可选的 `make start-lan` 仅供受信任的家庭/实验室局域网使用，要求 Basic Auth，并将防火墙范围限制为本地子网。它不是多用户权限系统；消费级 GPU 与 Docker 不提供公网多租户所需的强 GPU/显存隔离。严禁公网、路由器端口转发或公共 Wi-Fi 暴露。

## Quick Start

前置条件：WSL2、Windows NVIDIA 驱动、Docker Desktop（启用当前 WSL 发行版集成）和 NVIDIA Container Toolkit。仅运行 Compose 不需要宿主 Node/NVCC；开发测试需要 Python 3.12+ 与 Node.js 20+。

```bash
git clone git@github.com:BabyDrangoner/MyLeetGpu.git
cd MyLeetGpu
cp .env.example .env
make doctor
make start
```

Windows 浏览器访问：<http://localhost:3000>

```bash
make logs          # 查看 API/Worker/Web 日志
make stop          # 停止服务，保留草稿、版本和 benchmark
make clean-jobs    # 清理不再被活动任务引用的临时目录
```

首次执行 `make doctor` 可能会拉取固定版本的 CUDA 和 PyTorch/Triton 镜像，耗时取决于网络。持久数据保存在 `./data/myleetgpu.db`；源码快照只会在用户显式“保存为性能版本”且完整验证、benchmark 均成功后进入数据库。

### CUDA C++、Triton 与 PyTorch

Vector Addition、Matrix Transpose、Sum Reduction、Max Reduction、Softmax、Matrix Multiplication、Top-K 和 Top-P 支持在编辑器顶部切换 `CUDA C++` 与 `Triton (Python)`；MHA 和 GQA 使用 `PyTorch (Python)`，要求实现保存固定投影权重的自注意力 class，唯一前向输入为 `X` 与 `isCasual`。URL、服务端草稿、本地回退草稿和性能版本都带实现语言；切换语言不会覆盖另一套源码。

Triton 的“编译”动作在无 GPU 容器中完成 Python 语法检查和 `restricted_triton_v2` 提交策略预检；只允许题面说明中的 `@triton.jit` 子集与直线式 `solve` launcher，文件/网络/进程、反射、动态执行和打印会被拒绝。`@triton.jit` Kernel 会在运行或验证的第一次 GPU 调用中按实际参数完成 JIT 专化，因此预检通过后仍可能出现 Triton 编译错误。Triton 运行环境固定为官方 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` 镜像的审计 digest，当前包含 Python 3.11、PyTorch 2.5.1 + CUDA 12.4 和 Triton 3.1；实际版本和镜像摘要以环境页及 `make doctor` 的探测结果为准。

PyTorch 的“编译”动作同样是无 GPU 语法与 `restricted_torch_v2` AST 策略预检。当前 MHA/GQA 提交分别定义题目指定的 `MultiHeadAttention` / `GroupedQueryAttention` 普通类，由平台注入固定只读权重，并只计时 `forward(X, isCasual)`；允许使用白名单中的 reshape/transpose、matmul、causal mask、softmax 和 repeat-interleave 等基础运算。策略不开放 `torch.nn` 或现成 scaled-dot-product attention，也拒绝文件、网络、进程、反射、动态执行、打印、forward 状态写入和原地输出逃逸。运行、完整验证和 benchmark 由可信 harness 在 GPU 0 的受控 CUDA stream 上完成。

Triton 与 PyTorch 共用固定的官方 PyTorch/Triton 镜像，但分别探测并保存环境快照。性能版本按语言分组，只允许同一语言的版本比较和统一重测；系统不会生成 CUDA C++、Triton 与 PyTorch 之间的跨语言 speedup。语言资料可参考 [Triton 官方安装说明](https://triton-lang.org/main/getting-started/installation.html)、[Vector Addition 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 和 [PyTorch 官方版本页](https://docs.pytorch.org/get-started/previous-versions/)。

### 可选：同一局域网访问

本机已使用 WSL mirrored networking 时，可从提升权限的 Windows/WSL 终端依次执行：

```bash
make lan-firewall   # 只允许 LocalSubnet 访问当前 LAN IP 的 TCP/3000
make start-lan      # 首次会创建并只显示一次随机密码
```

终端会打印类似 `http://192.168.31.106:3000` 的地址。用户名默认为 `myleetgpu`。服务仍同时保留 `http://localhost:3000`，但 LAN overlay 启用期间两者都需要认证。轮换密码使用 `make lan-password`，关闭后执行 `make stop-lan` 和 `make lan-firewall-off`。不要发布 API 8000。

## 开发与测试

```bash
make install
make lint
make test
make test-gpu      # 必须连接真实 NVIDIA GPU，不会回退到 mock
make e2e
```

常用命令还包括 `make migrate`、`make recover-runner` 和 `make help`。默认 Compose 只绑定 loopback；可选 LAN overlay 只额外发布经过认证的 Web 端口，API 始终通过同源 `/api` 提供。

## 文档

- [系统设计](docs/design.md)
- [用户指南](docs/user-guide.md)
