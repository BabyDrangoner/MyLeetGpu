# 架构与维护指南

本文说明当前代码的职责分配与扩展入口。产品协议、安全边界、数据模型见 [系统设计](design.md)，启动和 Colab 连接见 [用户指南](user-guide.md)。

## 1. 架构选择

项目采用模块化单体：Web、API、Worker 分进程运行，共享同一题目协议和 SQLite 数据。保留单 Worker 串行执行，不引入消息中间件、微服务、DI 容器或新的前端状态框架。

采用的模式服务于已有变化点：

| 变化点 | 设计方式 | 作用 |
| --- | --- | --- |
| 新增执行设备 | Runner Protocol + 平级适配器 + 显式路由 | 判题用例不依赖 Docker/SSH；不允许静默回退 |
| 增加 HTTP 资源 | APIRouter + 类型化依赖 + Presenter | 路由、资源构建、错误映射、响应投影各有归属 |
| 调整判题规则 | JobExecutor 用例服务 + 纯结果函数 | 调度不包含判题细节；正确性和统计规则可独立测试 |
| 后端响应变化 | 前端 mapper 防腐层 | 兼容字段只在边界转换，组件使用统一领域类型 |
| 草稿与路由变化 | 以题目/语言为键的 DraftSession | 延迟请求、自动保存、离页 flush 不串写其他草稿 |
| 保存与重测 | Repository 的显式事务操作 | 完整验证与测量成功后才原子提交版本/批量测量 |

这不是强制全层 DTO/Repository 接口的 Clean Architecture。应用服务仍可依赖具体 Repository 和持久化记录；ORM 事务不能进入应用服务，ORM 对象不能直接作为 HTTP 响应。当前只有一个 SQLite 实现，没有为了假想数据库替换而复制全部模型。

## 2. 依赖与调用方向

```text
页面 / 展示组件
    ├─ 专属 hooks：草稿、工作区会话、任务轮询、布局、执行设置
    └─ api/client（组合门面）
           └─ resources → transport + 纯 mappers
                            │ HTTP /api
FastAPI main（组合、生命周期、中间件、错误处理）
    └─ routers → dependencies → 应用服务 / Repository
                                   │ SQLite 队列
Worker（领取、租约、目标选择、终态、finally 清理）
    └─ JobExecutor（编译、运行、验证、保存、重测）
           ├─ results（诊断映射、内部用例脱敏、benchmark 规则）
           ├─ Repository（短事务、原子保存）
           └─ Runner Protocol
                  └─ ExecutionRouter
                         ├─ DockerRunner ─┐
                         └─ ColabRunner  ─┴─ BaseRunner（共享机械逻辑）
```

Domain 不依赖 HTTP、数据库、Worker 或执行适配器。Application 可以依赖 Runner 协议和不可变结果类型，不能导入 DockerRunner、ColabRunner 或启动子进程。进程入口负责选择具体实现。

## 3. 后端代码导航

### API

- `backend/myleetgpu/api/main.py`：应用组合与进程入口，保留 `create_app` 和 ASGI `app`。
- `api/routers/`：health、problems、drafts、jobs、versions、environment 六个资源模块；执行位置设置归属 environment。
- `api/dependencies.py`：请求依赖提供者，测试可通过 FastAPI dependency override 替换用例或数据访问。
- `api/presenters.py`：显式响应投影，避免暴露源码临时路径、内部测试及其他 ORM 字段。
- `api/errors.py`：统一异常到 HTTP 错误 envelope 的映射。
- `api/runtime_status.py`：环境快照可用性与熔断状态读取，不执行 GPU 探测。

API 启动失败、正常退出和异常退出均释放数据库引擎。路由不编译、不执行用户代码；连接测试只是入队并异步等待 Worker。

### 调度与判题

- `worker.py`：宿主循环。维护唯一租约、领取任务/探测、选择目标、调用用例、处理成功/失败终态和清理。`create_worker()` 是资源上下文管理器，应使用 `with create_worker() as worker:`；CLI 入口保持 `python -m myleetgpu.worker`。
- `application/jobs.py`：提交校验、不可变源码快照、目标快照、重复版本检测、入队与孤立 spool 清理。
- `application/judging.py`：`JobExecutor.execute(job, spool)` 执行已领取任务。通过注入的 `check_lease` 回调在 GPU 阶段前检查租约，不知道线程、轮询或 HTTP。
- `application/results.py`：将执行结果转为安全业务结果或 `JobFailed`；检查 benchmark 的协议、规模、顺序和样本数，从原始样本重新计算统计。
- `application/compare.py`：最新 benchmark 选择、同语言/同环境可比性、逐规模 speedup。
- `infrastructure/repository.py`：SQL 与事务唯一归属；应用服务通过查询方法获取活动 spool 路径，不自行打开 Session。

`JobExecutor` 不领取任务、不更新最终成功状态、不销毁 spool；这些生命周期职责只属于 Worker。失败的保存不能创建版本；重测须全部验证、全部测量成功后一次性提交测量批次。

### Runner

- `runner/protocols.py`：结构化 `Runner` 端口；`TargetRunner` 表达可选的多目标选择/探测能力。测试替身无需继承实际适配器。
- `runner/router.py`：显式转发编译、执行、清理、探测、健康等操作，没有任意 `__getattr__` 转发。所有目标共享单 Worker，路由只在任务边界切换。
- `runner/base.py`：传输无关的源码准备、语言策略、结果解析、受限诊断、本地临时文件清理和健康文件写入。它不是可直接运行的默认执行器，不创建 Docker/SSH 连接。
- `runner/docker.py`、`runner/colab.py`：平级适配器，各自负责实际执行、工具链探测、超时、进程/容器清理与健康边界。Colab 不导入或继承 DockerRunner。
- `runner/colab_bridge.py`：远端标准库协议端点，保持独立可发送，不依赖本机安装整个项目。
- `runner/cpu.py`、`runner/cpu_process.py`：普通 C++/Python 本机适配器和子进程限制包装器，不调用 Docker/SSH/GPU。CPU 任务固定路由到 `cpu`，不使用 GPU 的全局执行设置；通过 `CpuEnvironmentRunner` 协议独立探测工具链，各语言有独立熔断标记。原生执行仅供可信单用户，不是文件或网络隔离沙箱。

本地和 Colab 的隔离不同；抽取共享代码不代表统一安全承诺。不得修改旧本地环境指纹算法来追求形式一致，避免已有性能记录失去可比性。

## 4. 前端代码导航

- `src/api/client.ts`：兼容门面与客户端组合；页面继续使用 `api`，测试/新调用方可以注入 transport 创建客户端。
- `src/api/transport.ts`：HTTP、错误、超时与外部取消组合。不关心题目、任务或版本字段。
- `src/api/resources/`：各资源端点与请求参数；不管理 React 状态。
- `src/api/mappers/`：纯 JSON → 领域模型转换；不发请求或访问浏览器状态。
- `src/hooks/useWorkspaceSession.ts`：当前题目身份与 URL 语言选择，避免新路由使用旧题 implementation。
- `src/hooks/useWorkspaceDraft.ts`：草稿读取、冲突优先级、编辑版本号、自动保存、pagehide/unmount flush；每个会话固定自己的题目、语言和源码。
- `src/pages/WorkspacePage.tsx`：组合布局、编辑器、动作与展示，不自行维护另一套草稿计时器。

任务提交快照、编辑中的草稿、服务器持久版本是三种不同状态，不共享可变对象。语言、题目或执行位置的变化不得改写已经提交的任务。

## 5. 如何扩展

### 新增题目

添加 `problems/<slug>` manifest、statement、各语言 starter/harness，并增加题目注册与协议测试。遵守 revision/suite hash 规则，无需在 Worker 或 API 写题目名称分支。

### 新增执行位置

实现 Runner 端口及适配器测试，在 ExecutionRouter 组合处注册；再显式扩展服务端允许的执行位置、前端类型/选择器、设置校验和环境指纹规则。不要开放网页任意主机、密钥或命令输入；不要给失败目标添加透明 fallback。新适配器必须测试断连、超时、清理、目标快照和“不误用另一设备”行为。

### 新增任务动作

修改领域动作及状态转换、提交校验、JobExecutor 分支和 API schema/前端操作。先明确动作是否允许持久化版本、是否需要 GPU、失败时保留什么；不要把业务写进 Worker 轮询循环。

### 修改响应格式

先更新 API presenter/schema 和契约测试，再修改前端 mapper/resource。对历史记录的兼容集中在这些边界，不把 snake_case、可选字段回退和时间解析分散到组件。

## 6. 验证与边界守卫

```bash
make lint
make test       # 排除本地及 Colab GPU 测试
make e2e        # 独立端口的浏览器回归
```

`tests/test_architecture.py` 用 AST 检查 Domain/Application 的禁止依赖和应用层直接 Session 访问。Runner 架构测试检查平级适配器、显式转发和不触发另一种传输。纯结果测试不需要 Worker 或 GPU；用例测试通过注入 Runner/租约检查验证顺序；Worker 测试覆盖状态机、原子保存和清理；API 测试覆盖 HTTP 与资源释放。

真实 GPU 验收仍须分别显式运行 `make test-gpu` 或 `make test-colab`。Colab 需现有会话和 SSH 主连接，验收前停止会竞争 GPU 的 Worker；不可将 CPU 替身测试或跳过项记作真实 GPU 成功。

## 7. 本轮刻意保留的边界

- 保留 SQLite 事务仓储，不引入通用 CRUD 基类、事件总线或第二份实体模型。
- 保留已验证的题目 harness、测量口径和存储 schema；结构重构本身不需要数据迁移。工作分支同时保留此前 Colab 功能所需的 `0006_execution_targets` 迁移。
- 保留单 Worker 串行模型，暂不做多 GPU 调度或跨设备并发。
- 保留现有页面视觉和交互，不将架构重构变成另一轮 UI 改版。
- 后续若出现第二个存储实现、多人并发编辑或多个独立执行队列，再分别评估仓储端口/DTO、乐观锁和调度模型；不要提前扩大本轮范围。
