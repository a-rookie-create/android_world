# Mobile-Agent-v2 AndroidWorld Adapter Notes

本文档备份最小修改方案：目标是让 Mobile-Agent-v2 作为 AndroidWorld 的
`EnvironmentInteractingAgent` 参与 benchmark，先跑通测试链路和指标产出，再逐步恢复
V2 的 memory、reflection、planning 等完整逻辑。

## AndroidWorld 测试链路

AndroidWorld 的测试入口在 `run.py`：

```text
run.py::_main()
  env_launcher.load_and_setup_env(...)
  suite_utils.create_suite(...)
  _get_agent(...)
  checkpointer_lib.create_run_directory(...)
  suite_utils.run(...)
  env.close()
```

其中真正需要接入 Mobile-Agent-v2 的位置是 `_get_agent()`。只要新增一个 agent
分支，后续的 task 初始化、episode 循环、成功判定、checkpoint 和汇总都可以继续使用
AndroidWorld 原生逻辑。

## Episode 头尾

每个 task 的执行在 `suite_utils._run_task()`：

```text
task.initialize_task(env)
interaction_results = run_episode(task)
task_successful = task.is_successful(env)
agent_successful = task_successful if interaction_results.done else 0.0
task.tear_down(env)
```

这意味着 Mobile-Agent-v2 不需要自己计算成功率。它只需要在认为任务完成时返回
`done=True`，AndroidWorld 会用 task 自带 validator 计算 `is_successful`。

每个 episode 的 step 循环在 `episode_runner.run_episode()`：

```text
agent.reset(start_on_home_screen)
agent.set_max_steps(max_n_steps)

for step_n in range(max_n_steps):
  result = agent.step(goal)
  output.append(result.data | {"step_number": step_n})
  if result.done:
    return EpisodeResult(done=True, step_data=...)
```

因此 V2 原本的 `while True` 需要拆成单步 `step(goal)`。

## 最小改动清单

第一阶段只做 AndroidWorld 原生适配，不改指标尾部。

需要新增：

```text
android_world/android_world/agents/mobile_agent_v2.py
```

需要修改：

```text
android_world/run.py
```

暂不需要修改：

```text
android_world/android_world/episode_runner.py
android_world/android_world/suite_utils.py
android_world/android_world/checkpointer.py
```

## Agent 骨架

`mobile_agent_v2.py` 应实现：

```python
class MobileAgentV2(base_agent.EnvironmentInteractingAgent):
  def __init__(self, env, ...):
    super().__init__(env, name="MobileAgentV2")
    self.history = ...
    self.memory = ...
    self.completed_requirements = ...
    self.error_flag = False

  def reset(self, go_home: bool = False):
    super().reset(go_home)
    self.env.hide_automation_ui()
    self.history = []
    self.memory = ""
    self.completed_requirements = ""
    self.error_flag = False

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    state = self.get_post_transition_state()
    perception_infos = ...
    prompt = ...
    model_output = ...
    action = ...
    converted_action = ...
    self.env.execute_action(converted_action)
    return base_agent.AgentInteractionResult(done=False, data=step_data)
```

这里应尽量贴近 AndroidWorld 自带 agent 的结构，而不是只把 V2 原脚本塞进来：

```text
M3A/T3A:
  reset() -> super().reset(...) -> env.hide_automation_ui() -> clear history
  step()  -> get_post_transition_state()
          -> build prompt from goal + history + UI elements
          -> parse action
          -> validate index
          -> env.execute_action(...)
          -> read after state
          -> summarize into history
          -> return AgentInteractionResult(done, step_data)

SeeAct:
  step()  -> get_post_transition_state()
          -> format/filter actionable elements
          -> generate action
          -> ground to element
          -> execute action
          -> return done when action_type == status
```

因此第一版 MobileAgentV2 也应该保留 AndroidWorld 示例里的几个约定：

```text
1. reset 时调用 env.hide_automation_ui()。
2. 每步先调用 get_post_transition_state()，不要直接 ADB 截图。
3. 每步保留 before state；执行后再取 after state。
4. 解析出 index 类动作后先做越界校验。
5. 每步必须生成 summary，哪怕先用简单模板，也要 append 到 self.history。
6. status/Stop 类动作返回 done=True。
```

`step_data` 中建议至少记录：

```text
raw_screenshot
before_ui_elements
after_screenshot
after_ui_elements
perception_infos
keyboard
action_prompt
action_raw_response
thought
action
operation_summary
converted_action
execution_error
memory
completed_requirements
```

注意：`step_data` 不要包含 `step_number`，AndroidWorld 会在
`episode_runner.run_episode()` 中自动添加。

## Perception 适配

V2 原版通过 ADB 自己截图，再跑 OCR 和 icon detection：

```text
adb screencap -> screenshot.jpg
ocr(...)
det(...)
caption icons
```

AndroidWorld 已经提供当前状态：

```python
state = self.get_post_transition_state()
state.pixels
state.ui_elements
```

第一版建议优先用 AndroidWorld 原生 state 构造 V2 的 `perception_infos`，避免绕过
benchmark 环境：

```text
UIElement.text/content_description/hint_text -> "text: ..."
UIElement bbox center                         -> [x, y]
```

构造时应参考 M3A/T3A 的 UI element 过滤逻辑。AndroidWorld 示例不是把所有节点都塞进
prompt，而是会通过 `m3a_utils.validate_ui_element(...)` 过滤掉不可见或无效元素，并保留
原始 index，这样模型输出的 index 才能回到 `state.ui_elements`。

V2 原版使用坐标动作，但 AndroidWorld 示例更推荐 index-based grounding。第一版可以同时
暴露两种信息：

```text
perception_infos: 给 V2 prompt 使用，形如 [x, y]; text/content
ui_elements:      给 action 执行和 index 校验使用，保持 AndroidWorld 原始列表
```

如果继续让 V2 输出 `Tap (x, y)`，则转成 coordinate click；如果后续把 prompt 改成
AndroidWorld JSON action 格式，则直接走 index action。

这样可以减少 ModelScope OCR、GroundingDINO、ADB 截图等额外依赖，先验证测试链路。
后续如需完全复现 V2 感知模块，再把 OCR/icon caption 接回来。

## Action 适配

V2 原始动作空间：

```text
Open app (app name)
Tap (x, y)
Swipe (x1, y1), (x2, y2)
Type (text)
Back
Home
Stop
```

AndroidWorld 动作类型在 `android_world.env.json_action.JSONAction` 中。

建议映射：

```text
Open app (app name)        -> JSONAction(action_type="open_app", app_name=...)
Tap (x, y)                 -> JSONAction(action_type="click", x=x, y=y)
Swipe (x1, y1), (x2, y2)   -> JSONAction(action_type="swipe", x=..., y=...)
Type (text)                -> JSONAction(action_type="input_text", text=..., index=...)
Back                       -> JSONAction(action_type="navigate_back")
Home                       -> JSONAction(action_type="navigate_home")
Stop                       -> AgentInteractionResult(done=True, data=step_data)
```

`Type` 是第一阶段最需要小心的动作。AndroidWorld 的 `input_text` 通常更适合
index-based editable field，而 V2 原版是假设键盘已激活后直接输入。第一版可以：

```text
1. 若当前存在 focused/editable UI element，则使用该 element index。
2. 若无法可靠定位输入框，则记录 execution_error，不执行输入。
3. 后续再加 fallback ADB text input。
```

还需要沿用 M3A/T3A 示例中的动作健壮性处理：

```text
1. parse 失败：不执行动作，写 summary，done=False。
2. index 越界：不执行动作，写 summary，done=False。
3. env.execute_action 抛异常：记录 execution_error，写 summary，done=False。
4. Stop/status：写 summary，done=True。
```

其中 `step_data["converted_action"]` 最好存 `JSONAction` 或其 `as_dict()` 结果，方便后续从
checkpoint 里复盘。

## run.py 注册方式

在 `android_world/run.py` 增加 import：

```python
from android_world.agents import mobile_agent_v2
```

在 `_get_agent()` 中增加：

```python
elif _AGENT_NAME.value == "mobile_agent_v2":
  agent = mobile_agent_v2.MobileAgentV2(env)
```

运行方式保持 AndroidWorld 原生 CLI：

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=mobile_agent_v2 \
  --tasks=ContactsAddContact \
  --n_task_combinations=1
```

## 第一阶段裁剪策略

为了先跑通 AndroidWorld 指标链路，第一版建议只保留：

```text
perception -> action prompt -> model output -> action parse -> execute -> done
```

但与 AndroidWorld 示例对齐后，第一版仍建议保留一个轻量 summary/history：

```text
summary = f"Action selected: {action}. {operation_summary or parser_status}"
self.history.append(step_data)
```

这不是 V2 完整 planning agent，而是为了让后续 step 能像 M3A/T3A 一样看到历史动作。

暂时裁掉：

```text
memory agent
reflection agent
planning/process agent
icon caption model
external OCR/detection
```

跑通后再逐步恢复：

```text
1. 加 memory。
2. 加 reflection。
3. 加 planning/completed_requirements。
4. 视需要恢复 V2 原始 OCR/icon detection。
```

## 与 AndroidWorld 示例的对照结论

本适配应以 M3A/T3A 的 agent 生命周期为主，而不是以 V2 原版 CLI 脚本为主。

需要直接继承的 AndroidWorld 示例设计：

```text
M3A/T3A 的 reset/history/summary 模式
M3A/T3A 的 get_post_transition_state() 状态获取
M3A/T3A 的 JSONAction 解析、index 校验、execute_action 错误处理
M3A 的 before/after screenshot 或 T3A 的 before/after element list 记录方式
SeeAct 的两段式“生成意图 -> grounding/action”的可扩展结构
```

暂时不需要继承的示例细节：

```text
M3A 的 set-of-mark 图片标注，除非后续模型确实需要。
M3A/T3A 的独立 summarization LLM call，第一版可用模板 summary。
SeeAct 的 OpenAI request 实现，模型 API 后续单独接。
MiniWoB additional_guidelines，除非跑 MiniWoB family。
```

## 指标与结果

无需改结果尾部。AndroidWorld 会在 `suite_utils.process_episodes()` 中按 task template
汇总：

```text
num_runs
mean_success_rate
mean_episode_length
mean_run_time
```

checkpoint 会写到 `--output_path` 下的 run 目录，每个 task instance 一个 `.pkl.gz`。

如果后续希望更容易分析 V2 行为，可以在 `step_data` 中额外记录 prompt、raw response、
parsed action、memory 等字段；这些字段会自然进入 episode checkpoint。
