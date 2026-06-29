# Mobile-Agent-v2 AndroidWorld Adapter

目标：用 AndroidWorld 评测 Mobile-Agent-v2，同时尽量保持被测 Agent 与原版 V2 一致。

## 核心判断

AndroidWorld 不匹配中间动作序列，也不要求 agent 使用某个标准动作空间。

一次任务是否成功，主要看 episode 结束后的设备状态：

```text
agent 真实操作设备
-> episode 结束
-> task.is_successful(env) 检查最终设备状态
-> 若状态满足任务要求，并且 agent 返回 done=True
-> 任务成功
```

所以适配重点不是把 V2 动作硬改成 AndroidWorld `JSONAction`，而是让 AndroidWorld 能按
`step()` 调用 V2，让 V2 继续真实操作当前 emulator。

## 分工

AndroidWorld 继续负责：

```text
task 初始化
episode step 循环
step budget
最终状态判分 task.is_successful(env)
checkpoint / 指标汇总
```

Mobile-Agent-v2 继续负责：

```text
截图 / OCR / icon caption / perception
planning / memory / reflection
动作选择
动作执行
完成判断
```

Adapter 只负责：

```text
实现 AndroidWorld Agent 接口
把 goal 传给 V2
让 V2 执行一轮 observe-decide-act
把 V2 的完成信号转成 done=True
记录 step_data
```

## 最小改动范围

新增：

```text
android_world/android_world/agents/mobile_agent_v2.py
```

修改：

```text
android_world/run.py
```

不改：

```text
android_world/android_world/episode_runner.py
android_world/android_world/suite_utils.py
android_world/android_world/task_evals/*
android_world/android_world/checkpointer.py
```

## 调用结构

```text
run.py
  -> suite_utils.run(...)
    -> episode_runner.run_episode(...)
      -> MobileAgentV2Wrapper.step(goal)
        -> 调用 V2 的一轮 observe-decide-act
        -> V2 自己操作 emulator
        -> 返回 AgentInteractionResult(done=..., data=...)
    -> task.is_successful(env)
```

V2 原脚本里的主循环通常类似：

```text
while True:
  observe
  decide
  execute
  update memory / reflection / planning
```

接入 AndroidWorld 后，应拆成：

```text
AndroidWorld episode loop 负责 while
MobileAgentV2Wrapper.step(goal) 只执行一轮 V2 行为
```

## Agent 外壳

```python
class MobileAgentV2Wrapper(base_agent.EnvironmentInteractingAgent):
  def reset(self, go_home: bool = False):
    super().reset(go_home)
    self.env.hide_automation_ui()
    self.v2_state.reset()

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = self.v2_runner.run_one_step(goal)
    return base_agent.AgentInteractionResult(
        done=step_data.done,
        data=step_data.to_dict(),
    )
```

`step()` 不应该重新实现 V2 的核心逻辑，只应该调用 V2 原有模块或从 V2 主循环中拆出的
单步函数。

## 设备交互

V2 可以保留自己的动作空间：

```text
Open app
Tap
Swipe
Type
Back
Home
Stop
```

执行动作时优先保持 V2 原语义。可以使用 V2 原 controller/ADB，也可以在必要时调用
AndroidWorld 的 `env.controller` / `adb_utils`。只有确实等价时，才把动作转成
`JSONAction`。

## done=True

AndroidWorld 最终成功要求两件事同时成立：

```text
task.is_successful(env) > 0.5
interaction_results.done == True
```

因此当 V2 输出 `Stop`、判断任务完成，或达到其内部完成条件时，adapter 必须返回：

```python
base_agent.AgentInteractionResult(done=True, data=step_data)
```

未完成时返回 `done=False`，由 AndroidWorld 继续下一步。

## step_data

建议记录：

```text
screenshot_path / raw_screenshot
v2_prompt
v2_raw_response
v2_action
v2_operation_summary
execution_result
execution_error
memory / reflection / planning 输出
```

不要写 `step_number`，AndroidWorld 会在 `episode_runner.run_episode()` 中自动添加。

## run.py 接入

真正实现 wrapper 后，在 `run.py` 增加：

```python
from android_world.agents import mobile_agent_v2

elif _AGENT_NAME.value == "mobile_agent_v2":
  agent = mobile_agent_v2.MobileAgentV2Wrapper(env)
```

运行示例：

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=mobile_agent_v2 \
  --tasks=ContactsAddContact \
  --n_task_combinations=1
```

## 快照 setup

快照与具体 agent 无关。首次 app setup / snapshot 使用：

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=t3a_gpt4 \
  --perform_emulator_setup \
  --setup_only
```

`--setup_only` 只做 app setup 和 snapshot 保存，不初始化 agent，不跑 benchmark。

