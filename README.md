# τ-bench evaluation + SFT + GRPO

真实结果优先，当前为阶段 1。**尚未产生 3B/7B 基线或训练结果。**

服务器根目录 `/traindata2/ganmuv3`，项目位于 `projects/agentic-grpo`。
共享账号 wangwei；Git 署名仅仓库局部配置为 jack。不要修改全局环境、删除其他进程或把 GitHub 凭据留在服务器。

## 当前复现入口

```bash
cd /traindata2/ganmuv3/projects/agentic-grpo
source scripts/activate.sh
python -m eval.catalog
python -m unittest discover -s tests -v
```

环境安装由专属 tmux 执行：

```bash
tmux -L ganmuv3-agentic new-session -d -s bootstrap 'bash scripts/bootstrap.sh'
tail -f logs/bootstrap.log
```

uv/Python、模型、环境、缓存和临时文件都限定在个人前缀；实际版本及哈希见 configs/*.lock、artifacts/*manifest* 和 EXPERIMENTS.md。

## 用户模拟器：本地代理模式

用户选择密钥只留在个人电脑进程内存，不写共享服务器。

个人电脑运行（隐藏输入 API key）：

```bash
python3 scripts/local_proxy.py --max-requests 8 --max-tokens 2048
```

另一终端建立连接：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:18761:127.0.0.1:18761 196
```

服务器只设无密钥配置：

```bash
export USER_SIM_BASE_URL=http://127.0.0.1:18761/v1
export USER_SIM_MODEL=deepseek-flash
export USER_SIM_API_KEY=local-proxy-no-upstream-secret
python -m eval.smoke_simulator
```

代理固定上游 api.deepseek.com，仅支持模拟器文本请求，禁用 thinking；初始预算只够少量连通测试。断开代理或隧道会使模拟器调用失败，必须记录，不能归因策略模型。DeepSeek seed 未有可用保证，代理不转发 seed；记录模型响应信息，无法承诺 API 位级复现。正式实验前需提高经用户确认的请求预算。

等待 GPU 时可只运行无密钥模型转发：

```bash
python3 -u scripts/local_proxy.py --model-only --daemon
bash scripts/start_tunnel.sh
```

该模式拒绝所有计费 API 请求，仅转发指定两款 Qwen 的官方公开文件；电脑需要保持联网和唤醒。代理 PID/日志与 SSH control socket 位于个人电脑的项目 logs/，`bash scripts/stop_tunnel.sh` 可停止隧道。下载在服务器 `tmux -L ganmuv3-agentic` 的 model-relay 会话执行，支持 HTTP Range 续传并核对上游文件哈希。重新开启模拟器前须停止模型专用代理，按上面的隐藏输入方式启动 API 模式。

## GPU 服务与评测（待 GPU 分配）

下载不需要 GPU：`python -m scripts.download_models`；推理环境安装：`bash scripts/install_vllm.sh`。长任务必须放入专属 tmux，日志写 logs/。

```bash
export APPROVED_GPU_UUIDS=GPU-指定的完整UUID
export POLICY_MODEL_PATH=/traindata2/ganmuv3/models/Qwen2.5-3B-Instruct/固定revision
export POLICY_MODEL=Qwen2.5-3B-Instruct
tmux -L ganmuv3-agentic new-session -d -s policy 'bash scripts/serve_policy.sh > logs/policy.log 2>&1'
export POLICY_BASE_URL=http://127.0.0.1:18762/v1
export POLICY_API_KEY=local-policy
python -m eval.run_eval --domain retail --split test --task-ids 0 1 --trials 4 \
  --concurrency 1 --output results/唯一实验名
python -m analysis.failure_taxonomy results/唯一实验名
```

正式 retail 基线省略任务切片参数；airline 使用 `--domain airline`。所有源码先 commit。恢复用同一命令追加 `--resume`，配置与代码版本必须完全一致。已记录的系统错误不会被静默重跑覆盖，需要另开实验。

## 指标口径

- 每题 N 次中成功 c 次，pass^k = 任务平均 C(c,k)/C(N,k)，k=1..4；**不是** pass@k。
- 完整计划未全部落盘时，不输出总体 pass^k；同时记录覆盖率和系统异常。完整实验主指标将系统异常计为非成功，并明确异常数。
- 轮数是策略模型生成次数，包括生成被截断的一次；工具次数为实际执行的工具调用数。遵循官方 TC 的只执行首个工具约定，原始多工具响应另行保留。
- token 取各客户端的真实 API usage 累计，含多轮重复历史；策略/模拟器分列。缺失标记缺失，不估算成本或 tokens。
- 规则无法证明的语义性失败归入“未确定”，不把 reward=0 自动解释为违反 policy。人工复核前保留待核对状态。
- 每条 trial 原子存储；进程锁禁止同时写同一 run；配置、任务/policy/tools 哈希和代码 revision 随结果保存。

## 版本与任务边界

使用原始 sierra-research/tau-bench，锁定上游 commit，不替换成新版 benchmark，不修改官方 reward。官方环境的已知实现问题另记在 EXPERIMENTS.md。测试脚本使用的合成响应只验证代码，不能放入基线表。

retail 与 airline 的实际任务划分以 `artifacts/task-catalog.json` 为准。test 不用于采样 SFT 或调参。airline 若缺官方 train/dev，则在后续训练前单独确定方案。

GitHub push 使用个人电脑认证；仓库远端 https://github.com/doublestrips/agentic-grpo.git。密钥、完整轨迹、模型和缓存默认不提交。只提交脱敏的实验摘要、小型配置、代码与复现说明。
