# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mobile-Agent-v2 adapter for AndroidWorld."""

from __future__ import annotations

import base64
from collections.abc import Sequence
import dataclasses
import importlib.util
import io
import os
from pathlib import Path
import re
from typing import Any

from absl import logging
from android_world.agents import base_agent
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
from PIL import Image
import requests


_MOBILE_AGENT_V2_ROOT = (
    Path(__file__).resolve().parents[3] / 'MobileAgent' / 'Mobile-Agent-v2'
)
_PROMPT_PATH = _MOBILE_AGENT_V2_ROOT / 'MobileAgent' / 'prompt.py'
if _PROMPT_PATH.exists():
  _PROMPT_SPEC = importlib.util.spec_from_file_location(
      'mobile_agent_v2_prompt',
      _PROMPT_PATH,
  )
  if _PROMPT_SPEC is None or _PROMPT_SPEC.loader is None:
    raise ImportError(f'Failed to load Mobile-Agent-v2 prompt from {_PROMPT_PATH}')
  _PROMPT_MODULE = importlib.util.module_from_spec(_PROMPT_SPEC)
  _PROMPT_SPEC.loader.exec_module(_PROMPT_MODULE)
  get_action_prompt = _PROMPT_MODULE.get_action_prompt
  _PROMPT_IMPORT_ERROR = None
else:
  get_action_prompt = None
  _PROMPT_IMPORT_ERROR = FileNotFoundError(_PROMPT_PATH)


_DEFAULT_ADD_INFO = (
    'If you want to tap an icon of an app, use the action "Open app". If you'
    ' want to exit an app, use the action "Home"'
)
DEFAULT_ADD_INFO = _DEFAULT_ADD_INFO


@dataclasses.dataclass(frozen=True)
class ParsedV2Action:
  """Parsed Mobile-Agent-v2 action."""

  action_type: str
  raw_action: str
  x: int | None = None
  y: int | None = None
  x2: int | None = None
  y2: int | None = None
  text: str | None = None
  app_name: str | None = None


def _extract_section(text: str, start: str, end: str | None = None) -> str:
  if start not in text:
    return ''
  content = text.split(start, 1)[1]
  if end and end in content:
    content = content.split(end, 1)[0]
  return content.replace('\n', ' ').replace('  ', ' ').strip()


def parse_v2_output(output: str) -> tuple[str, str, str]:
  """Extracts thought, action, and operation from a V2 model response."""
  thought = _extract_section(output, '### Thought ###', '### Action ###')
  action = _extract_section(output, '### Action ###', '### Operation ###')
  operation = _extract_section(output, '### Operation ###')
  return thought, action, operation


def parse_v2_action(action: str) -> ParsedV2Action:
  """Parses Mobile-Agent-v2's text action format."""
  action = action.strip()

  match = re.fullmatch(
      r'(?i)\s*Open app\s*\((.*?)\)\s*(?::\s*(.*?))?\s*',
      action,
  )
  if match:
    app_name = match.group(2) or match.group(1)
    if app_name.strip().lower() == 'app name':
      app_name = ''
    return ParsedV2Action(
        action_type='open_app',
        raw_action=action,
        app_name=app_name.strip().strip('"'),
    )

  match = re.fullmatch(r'(?i)\s*Tap\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*', action)
  if match:
    return ParsedV2Action(
        action_type='click',
        raw_action=action,
        x=int(match.group(1)),
        y=int(match.group(2)),
    )

  match = re.fullmatch(
      r'(?i)\s*Swipe\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*'
      r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*',
      action,
  )
  if match:
    return ParsedV2Action(
        action_type='swipe',
        raw_action=action,
        x=int(match.group(1)),
        y=int(match.group(2)),
        x2=int(match.group(3)),
        y2=int(match.group(4)),
    )

  match = re.fullmatch(r'(?is)\s*Type\s*\((.*)\)\s*', action)
  if match:
    text = match.group(1).strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
      text = text[1:-1]
    return ParsedV2Action(
        action_type='input_text',
        raw_action=action,
        text=text,
    )

  if re.fullmatch(r'(?i)\s*Back\s*', action):
    return ParsedV2Action(action_type='navigate_back', raw_action=action)

  if re.fullmatch(r'(?i)\s*Home\s*', action):
    return ParsedV2Action(action_type='navigate_home', raw_action=action)

  if re.fullmatch(r'(?i)\s*Stop\s*', action):
    return ParsedV2Action(action_type='status', raw_action=action)

  raise ValueError(f'Unsupported Mobile-Agent-v2 action: {action}')


def _image_to_data_url(pixels: Any) -> str:
  image = Image.fromarray(pixels).convert('RGB')
  buffer = io.BytesIO()
  image.save(buffer, format='JPEG')
  encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
  return f'data:image/jpeg;base64,{encoded}'


def _element_text(element: representation_utils.UIElement) -> str:
  parts = [
      element.text,
      element.content_description,
      element.hint_text,
      element.resource_name,
      element.tooltip,
  ]
  return ' | '.join(part for part in parts if part)


def ui_elements_to_perception_infos(
    ui_elements: Sequence[representation_utils.UIElement],
) -> list[dict[str, Any]]:
  """Converts AndroidWorld UI elements to V2 perception_infos."""
  perception_infos = []
  for element in ui_elements:
    text = _element_text(element)
    if not text or element.bbox_pixels is None:
      continue
    x, y = element.bbox_pixels.center
    perception_infos.append({
        'text': 'text: ' + text,
        'coordinates': [int(x), int(y)],
    })
  return perception_infos


def _keyboard_active(
    ui_elements: Sequence[representation_utils.UIElement],
    height: int,
) -> bool:
  keyboard_y_min = 0.75 * height
  for element in ui_elements:
    text = _element_text(element)
    if 'ADB Keyboard' in text:
      return True
    if element.is_focused and element.is_editable:
      return True
    if element.bbox_pixels and element.bbox_pixels.y_min >= keyboard_y_min:
      if 'keyboard' in text.lower():
        return True
  return False


class MobileAgentV2(base_agent.EnvironmentInteractingAgent):
  """Runs Mobile-Agent-v2 style decisions inside AndroidWorld."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      api_url: str | None = None,
      token: str | None = None,
      model: str = 'gpt-4o',
      add_info: str = _DEFAULT_ADD_INFO,
      request_timeout: float = 120.0,
  ):
    super().__init__(env, name='MobileAgentV2')
    self.api_url = api_url or os.environ.get('MOBILE_AGENT_V2_API_URL', '')
    self.token = token or os.environ.get('MOBILE_AGENT_V2_API_TOKEN', '')
    self.model = model or os.environ.get('MOBILE_AGENT_V2_MODEL', 'gpt-4o')
    self.add_info = add_info
    self.request_timeout = request_timeout
    self.thought_history: list[str] = []
    self.summary_history: list[str] = []
    self.action_history: list[str] = []
    self.completed_requirements = ''
    self.memory = ''
    self.last_summary = ''
    self.last_action = ''
    self.error_flag = False

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home)
    self.env.hide_automation_ui()
    self.thought_history = []
    self.summary_history = []
    self.action_history = []
    self.completed_requirements = ''
    self.memory = ''
    self.last_summary = ''
    self.last_action = ''
    self.error_flag = False

  def _call_model(self, prompt: str, pixels: Any) -> tuple[str, Any]:
    if not self.api_url:
      raise ValueError(
          'Mobile-Agent-v2 API URL is empty. Set --mobile_agent_v2_api_url or'
          ' MOBILE_AGENT_V2_API_URL.'
      )

    content = [
        {'type': 'text', 'text': prompt},
        {'type': 'image_url', 'image_url': {'url': _image_to_data_url(pixels)}},
    ]
    messages = [
        {
            'role': 'system',
            'content': [{
                'type': 'text',
                'text': (
                    'You are a helpful AI mobile phone operating assistant.'
                    " You need to help me operate the phone to complete the"
                    " user's instruction."
                ),
            }],
        },
        {'role': 'user', 'content': content},
    ]
    headers = {'Content-Type': 'application/json'}
    if self.token:
      headers['Authorization'] = f'Bearer {self.token}'
    data = {
        'model': self.model,
        'messages': messages,
        'max_tokens': 2048,
        'temperature': 0.0,
        'seed': 1234,
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        self.api_url,
        headers=headers,
        json=data,
        timeout=self.request_timeout,
    )
    response.raise_for_status()
    response_json = response.json()
    return response_json['choices'][0]['message']['content'], response_json

  def _to_json_action(
      self,
      parsed_action: ParsedV2Action,
  ) -> json_action.JSONAction | None:
    if parsed_action.action_type == 'swipe':
      return None
    if parsed_action.action_type == 'status':
      return json_action.JSONAction(
          action_type=json_action.STATUS,
          goal_status='task_complete',
      )
    if parsed_action.action_type == 'open_app':
      return json_action.JSONAction(
          action_type=json_action.OPEN_APP,
          app_name=parsed_action.app_name,
      )
    if parsed_action.action_type == 'click':
      return json_action.JSONAction(
          action_type=json_action.CLICK,
          x=parsed_action.x,
          y=parsed_action.y,
      )
    if parsed_action.action_type == 'input_text':
      return None
    return json_action.JSONAction(action_type=parsed_action.action_type)

  def _execute_parsed_action(
      self,
      parsed_action: ParsedV2Action,
  ) -> json_action.JSONAction | str:
    converted_action = self._to_json_action(parsed_action)
    if parsed_action.action_type == 'swipe':
      command = adb_utils.generate_swipe_command(
          parsed_action.x,
          parsed_action.y,
          parsed_action.x2,
          parsed_action.y2,
          500,
      )
      adb_utils.issue_generic_request(command, self.env.controller)
      return ' '.join(command)

    if parsed_action.action_type == 'input_text':
      adb_utils.type_text(parsed_action.text or '', self.env.controller)
      return f'type_text {parsed_action.text or ""}'

    if converted_action.action_type == json_action.STATUS:
      return converted_action

    self.env.execute_action(converted_action)
    return converted_action

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = {
        'raw_screenshot': None,
        'ui_elements': [],
        'perception_infos': [],
        'action_prompt': None,
        'action_raw_response': None,
        'thought': None,
        'action': None,
        'operation': None,
        'converted_action': None,
        'summary': None,
        'execution_error': None,
    }
    logging.info('----------Mobile-Agent-v2 step %s----------',
                 str(len(self.action_history) + 1))

    state = self.get_post_transition_state()
    ui_elements = state.ui_elements
    height, width = state.pixels.shape[:2]
    perception_infos = ui_elements_to_perception_infos(ui_elements)
    keyboard = _keyboard_active(ui_elements, height)

    step_data['raw_screenshot'] = state.pixels.copy()
    step_data['ui_elements'] = ui_elements
    step_data['perception_infos'] = perception_infos

    if get_action_prompt is None:
      step_data['execution_error'] = (
          'Failed to import Mobile-Agent-v2 prompt.py from'
          f' {_MOBILE_AGENT_V2_ROOT}: {_PROMPT_IMPORT_ERROR}'
      )
      step_data['summary'] = 'Failed to import Mobile-Agent-v2 prompt.py.'
      return base_agent.AgentInteractionResult(False, step_data)

    prompt = get_action_prompt(
        goal,
        perception_infos,
        width,
        height,
        keyboard,
        self.summary_history,
        self.action_history,
        self.last_summary,
        self.last_action,
        self.add_info,
        self.error_flag,
        self.completed_requirements,
        self.memory,
    )
    step_data['action_prompt'] = prompt

    try:
      output, raw_response = self._call_model(prompt, state.pixels)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Failed to call Mobile-Agent-v2 model.')
      step_data['execution_error'] = str(e)
      step_data['summary'] = 'Failed to call Mobile-Agent-v2 model.'
      return base_agent.AgentInteractionResult(False, step_data)

    step_data['action_raw_response'] = raw_response
    thought, action, operation = parse_v2_output(output)
    step_data['thought'] = thought
    step_data['action'] = action
    step_data['operation'] = operation

    if not action:
      step_data['summary'] = (
          'Output for action selection is not in the correct V2 format, so no'
          ' action is performed.'
      )
      step_data['execution_error'] = output
      return base_agent.AgentInteractionResult(False, step_data)

    try:
      parsed_action = parse_v2_action(action)
      converted_action = self._execute_parsed_action(parsed_action)
      step_data['converted_action'] = converted_action
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Failed to parse or execute Mobile-Agent-v2 action.')
      step_data['execution_error'] = str(e)
      step_data['summary'] = (
          'Can not parse or execute the Mobile-Agent-v2 action. Make sure the'
          ' action follows the V2 action format.'
      )
      self.error_flag = True
      self.last_summary = operation
      self.last_action = action
      return base_agent.AgentInteractionResult(False, step_data)

    step_data['summary'] = operation
    self.thought_history.append(thought)
    self.summary_history.append(operation)
    self.action_history.append(action)
    self.last_summary = operation
    self.last_action = action
    self.error_flag = False

    done = parsed_action.action_type == 'status'
    return base_agent.AgentInteractionResult(done, step_data)
