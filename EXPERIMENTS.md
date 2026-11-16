# 实验记录

## 2026-11-16：阶段 1 初始化

- 状态：进行中；尚无策略模型评测结果，不填基线分数。
- 服务器：196 / thorough196，共享账号 wangwei。
- 个人前缀：`/traindata2/ganmuv3`；项目：`projects/agentic-grpo`。
- Git 局部署名：jack <wanxueyu1999@163.com>；远端：doublestrips/agentic-grpo。
- 上游 tau-bench：`59a200c6d575d595120f1cb70fea53cef0632f6b`。
- 上游目录：`/traindata2/ganmuv3/sources/tau-bench`，不直接修改任务。
- 本次检查：目标目录原为空；NFS 挂载可用 12T、使用率 98%。8 张 GPU 全有驻留进程，尚未获分配，不启动 GPU 服务。
- 当前非交互 SSH 环境缺少三个 USER_SIM_* 变量，未发起计费 API 请求。
- 后续用户提供 DeepSeek 兼容 API 配置并选择本地代理；真实 key 仅在个人电脑代理内存，不存服务器。服务器配置 localhost 转发地址与无密钥占位字符串。
- 长任务：个人 tmux socket `ganmuv3-agentic`，日志 `logs/`；停止操作只针对本项目已记录的会话/PID。

### 踩坑

1. Git 2.25.1 在首次 commit 前执行 `git branch -m main` 报 `refname refs/heads/master not found`。改用 `git symbolic-ref HEAD refs/heads/main` 设置 unborn branch。局部 user.name/email 和 remote 均验证成功。
2. `nvcc` 不在 PATH，不能因此断定未安装 CUDA。绝对路径确认 Toolkit 12.2.91；nvidia-smi 的 CUDA 12.4 是驱动兼容信息。选择 runtime wheel 时需分开核查。
3. 本地 gh 已有认证失效；不改共享服务器认证，不声称已 push。后续需用户完成本地 GitHub 认证。
4. 首次依赖解析失败：litellm 1.68.1 要求 openai >=1.68.2,<1.76.0，与初选 openai 1.78.1 冲突。将 openai 固定为 1.75.0 后重新解析；不绕过 resolver 强装。
5. 修正后独立 eval 环境安装退出码 0，Python 3.11.14；openai/litellm/numpy/pydantic/官方 get_env 导入通过。
6. 首次模拟器请求返回本地代理 502。公开端点诊断证明 Homebrew Python 默认 CA 验证失败（self-signed certificate in certificate chain）。使用系统 `/etc/ssl/cert.pem` 且保持 TLS 校验后，真实 DeepSeek 双轮模拟器测试成功：初始消息询问营业时间，答复 9 AM 后返回 `###STOP###`。这不是 benchmark 成绩。原失败与重试结果分别保存在 results/simulator-smoke/ 和 results/simulator-smoke-ca-fix/。
7. 服务器直连 huggingface.co 报 Network is unreachable / 连接超时；备用源亦解析超时。使用本地只读 HF 转发，远端文件直接写服务器，不在本地保存模型副本；以官方模型元数据的 immutable revision、LFS SHA256 / Git blob ID 核验。
8. 共享服务器落盘真实 API key 的自动审批被拒绝。用户随后选择本地代理方案；确认服务器不存在 config/user_sim.env 真实密钥文件。

### 已执行验证

- `python -m eval.catalog`：retail train=500、dev=20、test=115；airline test=50，train/dev 不存在。
- `python -m unittest discover -s tests -v` 首轮 9 项通过（1.414s），为离线合成响应/真实环境验证，不是模型评测。
- 增加 CLI 端到端并发、恢复不重复请求、拒绝配置变化、仅补齐缺失 trial 后，10 项测试全部通过（41.266s）。
- 官方 TC 对照：使用相同脚本响应，独立评测与官方 solver 的 terminal reward 相同，执行工具数和轮数符合预期。
- 尚未启动 GPU 服务；官方模型示例、3B/7B retail 基线和真实失败分析待可用 GPU。

### 版本与复现限制

DeepSeek API 的 seed 行为未获保证，代理不转发 seed 并显式禁用 thinking；不宣称远端 API 位级可复现。保留每次请求的 usage、响应 model、system_fingerprint（如有）和完整响应。API 成本未实测，不估算金额。

## 指标纪律

合成测试只验证实现，不计入基线。所有真实评测以官方 reward 为依据；任务异常、模拟器异常和运行缺失单列。未运行的结果为 unavailable，禁止估算。
