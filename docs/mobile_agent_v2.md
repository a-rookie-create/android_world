# 用 AndroidWorld 评测 Mobile-Agent-v2

目标：在不改写 Mobile-Agent-v2 核心行为逻辑的前提下，用 AndroidWorld 的任务、episode
管理和最终状态判分能力，对 Mobile-Agent-v2 做 benchmark。

## 核心判断

AndroidWorld 不匹配中间动作序列，也不要求 agent 使用某个标准动作空间。一次任务是否成功，
主要看 episode 结束后的设备状态：

```text
agent 真实操作设备
-> episode 结束
-> task.is_successful(env) 检查最终设备状态
-> 若状态满足任务要求，并且 agent 返回 done=True
-> 任务成功
```

因此评测 Mobile-Agent-v2 时，不需要把它的每一步动作翻译成 AndroidWorld 的标准动作，
也不需要让 AndroidWorld 接管中间操作。Mobile-Agent-v2 可以继续按自己的方式观察、规划、
点击、滑动、输入和判断完成；AndroidWorld 只在最外层提供 episode 壳，并在结束后检查最终
设备状态是否满足任务要求。

## 总体方案

在 Mobile-Agent-v2 外面套一层 AndroidWorld agent 壳：

```text
AndroidWorld 初始化任务和模拟器状态
-> AndroidWorld 进入 episode loop
-> 每个 step 调用 Mobile-Agent-v2 执行一轮真实设备操作
-> Mobile-Agent-v2 直接操作同一个 emulator
-> Mobile-Agent-v2 判断是否完成，并通过 done=True/False 告诉 AndroidWorld
-> episode 结束后 AndroidWorld 调用 task.is_successful(env)
-> AndroidWorld 汇总成功率、轨迹、checkpoint 和指标
```

这里的壳不是动作 adaptor。它不负责理解、改写或校验 Mobile-Agent-v2 的动作序列，只负责
把 AndroidWorld 的任务目标传给 Mobile-Agent-v2，并把 Mobile-Agent-v2 的完成信号返回给
AndroidWorld。

## 分工边界

AndroidWorld 负责：

```text
任务采样和初始化
app / snapshot setup
episode step 循环
step budget / 超时控制
最终状态判分 task.is_successful(env)
checkpoint / metrics / result aggregation
```

Mobile-Agent-v2 负责：

```text
截图获取
OCR / icon caption / perception
planning / memory / reflection
动作选择
动作执行
任务完成判断
```

关键点是：中间的一切设备操作都由 Mobile-Agent-v2 包办。AndroidWorld 不参与 action
translation，也不要求 Mobile-Agent-v2 使用 `JSONAction`。

## 调用结构

建议接入后的结构如下：

```text
run.py
  -> suite_utils.run(...)
    -> episode_runner.run_episode(...)
      -> MobileAgentV2Agent.step(goal)
        -> 调用 Mobile-Agent-v2 的一轮 observe-decide-act
        -> Mobile-Agent-v2 直接操作 emulator
        -> 返回 AgentInteractionResult(done=..., data=...)
    -> task.is_successful(env)
```

AndroidWorld 的 `episode_runner.run_episode(...)` 仍然是外层循环。Mobile-Agent-v2 原来的
主循环通常类似：

```text
while not finished:
  observe
  decide
  execute
  update memory / reflection / planning
```

接入 AndroidWorld 后，应拆成：

```text
AndroidWorld episode loop 负责 while
Mobile-Agent-v2 agent.step(goal) 只执行一轮 observe-decide-act
```

这样 AndroidWorld 可以继续控制最大步数、记录每步信息，并在 agent 返回 `done=True` 或
达到 step budget 后统一结束 episode。

## AndroidWorld 壳

新增一个 AndroidWorld agent 类，例如：

```text
android_world/android_world/agents/mobile_agent_v2.py
```

职责很薄：

```python
class MobileAgentV2Agent(base_agent.EnvironmentInteractingAgent):
  def reset(self, go_home: bool = False):
    super().reset(go_home)
    self.env.hide_automation_ui()
    self.v2_runner.reset()

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = self.v2_runner.run_one_step(goal)
    return base_agent.AgentInteractionResult(
        done=step_data.done,
        data=step_data.to_dict(),
    )
```

这个类不应该重新实现 Mobile-Agent-v2 的 perception、planning、memory、reflection 或动作
策略。正确做法是复用 Mobile-Agent-v2 现有模块，或者从 Mobile-Agent-v2 原主循环中抽出
`run_one_step(goal)` 这一层。

## 设备交互方式

Mobile-Agent-v2 直接操作 AndroidWorld 启动并管理的同一个 emulator。它可以继续保留自己的
动作语义，例如：

```text
Open app
Tap
Swipe
Type
Back
Home
Stop
```

执行动作时优先保持 Mobile-Agent-v2 原有 controller / ADB 路径。只要它操作的是同一个
AndroidWorld env 对应的设备，AndroidWorld 就能在 episode 结束后读取最终设备状态并判分。

只有在工程实现上确实更方便、且语义完全等价时，才考虑复用 `env.controller` 或
`adb_utils`。不需要把 Mobile-Agent-v2 的动作转换成 AndroidWorld 的 `JSONAction`，也不需要
构造一个动作空间 adaptor。

## done=True 的含义

AndroidWorld 最终成功通常要求两件事同时成立：

```text
task.is_successful(env) > 0.5
最后一次 interaction_result.done == True
```

所以 Mobile-Agent-v2 判断任务完成时，壳必须把完成信号传回 AndroidWorld：

```python
base_agent.AgentInteractionResult(done=True, data=step_data)
```

典型触发条件包括：

```text
Mobile-Agent-v2 输出 Stop
Mobile-Agent-v2 内部判断任务已经完成
Mobile-Agent-v2 达到自己的完成条件并决定停止
```

未完成时返回 `done=False`，AndroidWorld 会继续下一步。若 Mobile-Agent-v2 一直不返回
`done=True`，即使最终设备状态已经满足任务要求，也可能因为 episode 没有正常完成而被判为
失败或超时。

## step_data 记录

`AgentInteractionResult.data` 只用于日志、debug 和后续分析，不参与 AndroidWorld 的最终
状态判分。建议记录：

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

不要在 `step_data` 里手动写 `step_number`。AndroidWorld 会在
`episode_runner.run_episode()` 中自动添加 step 信息。

## 最小改动范围

建议新增：

```text
android_world/android_world/agents/mobile_agent_v2.py
```

建议修改：

```text
android_world/run.py
```

尽量不改：

```text
android_world/android_world/episode_runner.py
android_world/android_world/suite_utils.py
android_world/android_world/task_evals/*
android_world/android_world/checkpointer.py
```

也就是说，AndroidWorld 的评测机制保持原样，只增加一个能调用 Mobile-Agent-v2 单步执行的
agent 实现，并在 `run.py` 中注册它。

## run.py 接入

真正实现 `MobileAgentV2Agent` 后，在 `run.py` 增加类似分支：

```python
from android_world.agents import mobile_agent_v2

elif _AGENT_NAME.value == "mobile_agent_v2":
  agent = mobile_agent_v2.MobileAgentV2Agent(env)
```

运行示例：

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=mobile_agent_v2 \
  --tasks=ContactsAddContact \
  --n_task_combinations=1
```


## 成功标准

这套方案成立的前提是：

```text
Mobile-Agent-v2 和 AndroidWorld 操作的是同一个 emulator
Mobile-Agent-v2 每次 step 只执行一轮动作，而不是自己跑完整个无限循环
Mobile-Agent-v2 完成任务时能返回 done=True
AndroidWorld episode 结束后能正常调用 task.is_successful(env)
```

满足这些条件后，Mobile-Agent-v2 不需要适配 AndroidWorld 的中间动作空间。评测的核心就是：
让 Mobile-Agent-v2 真实完成任务，然后让 AndroidWorld 检查最终设备状态。
