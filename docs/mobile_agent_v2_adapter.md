# Mobile-Agent-v2 AndroidWorld Adapter

目标：在尽量少改代码的前提下，把原 Mobile-Agent-v2 封装成 AndroidWorld
可测试的 agent，并尽量保持被测 agent 的行为与原 V2 一致。

核心原则：

```text
不改 AndroidWorld 测试框架。
不重写 Mobile-Agent-v2 决策逻辑。
只新增一层 adapter，负责输入输出转换。
```

## 整体结构

```text
AndroidWorld runner
  -> MobileAgentV2Adapter.step(goal)
      -> 把 AndroidWorld state 转成 V2 输入
      -> 调用 V2 风格的决策逻辑
      -> 把 V2 动作转成 AndroidWorld 可执行动作
      -> env.execute_action(...) 或必要的 controller ADB 调用
      -> 返回 AgentInteractionResult
```

AndroidWorld 继续负责：

```text
task 初始化
episode step 循环
step budget
成功率计算 task.is_successful(env)
checkpoint
结果汇总
```

Mobile-Agent-v2 继续负责：

```text
根据任务目标和当前屏幕决定下一步动作
维护 thought/action/summary history
后续恢复 memory/reflection/planning
```

Adapter 负责：

```text
goal/state/ui_elements/screenshot -> V2 输入
V2 输出动作 -> AndroidWorld action 或 done=True
V2 step 日志 -> AndroidWorld step_data
```

## 最小改动

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
android_world/android_world/checkpointer.py
```

## run.py 接入

在 `android_world/run.py` 中新增 import：

```python
from android_world.agents import mobile_agent_v2
```

在 `_get_agent()` 中新增：

```python
elif _AGENT_NAME.value == "mobile_agent_v2":
  agent = mobile_agent_v2.MobileAgentV2(env)
```

运行方式：

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=mobile_agent_v2 \
  --tasks=ContactsAddContact \
  --n_task_combinations=1
```

## Agent 接口

`MobileAgentV2` 必须实现 AndroidWorld 标准接口：

```python
class MobileAgentV2(base_agent.EnvironmentInteractingAgent):
  def reset(self, go_home: bool = False):
    super().reset(go_home)
    self.env.hide_automation_ui()
    self.history = []

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    ...
```

`step()` 每次只执行一步，不能保留 V2 原来的 `while True`。

## 输入转换

原 V2 输入来自：

```text
ADB screenshot
OCR
icon detection
caption
instruction
history
```

AndroidWorld 适配后优先来自：

```python
state = self.get_post_transition_state()
pixels = state.pixels
ui_elements = state.ui_elements
```

第一版用 `ui_elements` 构造 V2 的 `perception_infos`：

```text
[x, y]; text/content_description/hint_text
```

这样可以先避免额外依赖 OCR、GroundingDINO、ADB 截图。后续如需更贴近原 V2，再恢复原感知模块。

## 输出转换

保留 V2 原动作空间：

```text
Open app (app name)
Tap (x, y)
Swipe (x1, y1), (x2, y2)
Type (text)
Back
Home
Stop
```

映射到 AndroidWorld：

```text
Open app (...) -> JSONAction(action_type="open_app", app_name=...)
Tap (x, y)     -> JSONAction(action_type="click", x=x, y=y)
Type (...)     -> JSONAction(action_type="input_text", text=..., index=...)
Back           -> JSONAction(action_type="navigate_back")
Home           -> JSONAction(action_type="navigate_home")
Stop           -> AgentInteractionResult(done=True, data=step_data)
```

`Swipe` 是唯一需要特别处理的动作：AndroidWorld 的 `JSONAction`
只支持方向型 `swipe/scroll`，不支持 V2 的起止坐标。为了尽量保持原 V2
一致，第一版 adapter 应直接通过 AndroidWorld controller 调用
`adb_utils.generate_swipe_command(x1, y1, x2, y2, duration_ms)` 执行坐标滑动。
如果后续改成方向型 prompt，再映射为 `JSONAction(action_type="scroll" 或
"swipe", direction=...)`。

`Type` 需要谨慎处理：第一版优先找当前 focused/editable UI element；找不到就记录错误，不强行输入。

解析失败、参数越界、动作执行异常时，参考 M3A/T3A：记录 `summary` 和
`execution_error`，返回 `AgentInteractionResult(done=False, data=step_data)`，
不要中断整个 AndroidWorld runner。

## 保持 V2 一致性

尽量保留：

```text
Thought / Action / Operation 输出格式
Open app / Tap / Swipe / Type / Back / Home / Stop 动作语义
thought_history
summary_history
action_history
一轮只产生一个动作
```

第一版可暂缓：

```text
memory agent
reflection agent
planning/process agent
原 OCR/icon detection/caption 流程
```

跑通 AndroidWorld 测试链路后，再逐步恢复这些模块。

## step_data

每步至少记录：

```text
raw_screenshot
ui_elements
perception_infos
action_prompt
action_raw_response
thought
action
operation
converted_action
summary
execution_error
```

不要手动写 `step_number`，AndroidWorld 会自动添加。

## 成功率与结果

Adapter 不计算成功率。

任务结束后 AndroidWorld 会执行：

```text
task.is_successful(env)
```

结果和 checkpoint 仍由原框架写入 `--output_path` 下的 run 目录。
