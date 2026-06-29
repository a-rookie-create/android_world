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

"""AndroidWorld shell for evaluating Mobile-Agent-v2.

This module intentionally does not adapt Mobile-Agent-v2 actions into
AndroidWorld JSON actions. Mobile-Agent-v2 directly operates the emulator via
ADB; AndroidWorld owns only the episode loop and final task-state judgement.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Any

from absl import logging
from android_world.agents import base_agent
from android_world.env import interface
from PIL import Image
from PIL import ImageDraw


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
_MOBILE_AGENT_V2_ROOT = os.path.join(
    _REPO_ROOT, 'MobileAgent', 'Mobile-Agent-v2'
)


def _env_bool(name: str, default: bool) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.lower() in ('1', 'true', 'yes', 'y', 'on')


def _install_tf_keras_legacy_layers_shim() -> None:
  """Expose tf_keras legacy layers at the path TensorFlow 2.19 expects."""
  try:
    legacy_layers = importlib.import_module('tf_keras.src.legacy_tf_layers')
  except ModuleNotFoundError:
    return

  sys.modules.setdefault('tf_keras.legacy_tf_layers', legacy_layers)
  for module_name in (
      'base',
      'convolutional',
      'core',
      'normalization',
      'pooling',
      'variable_scope_shim',
  ):
    source_name = f'tf_keras.src.legacy_tf_layers.{module_name}'
    target_name = f'tf_keras.legacy_tf_layers.{module_name}'
    sys.modules.setdefault(target_name, importlib.import_module(source_name))


@dataclasses.dataclass
class MobileAgentV2Config:
  """Runtime configuration for Mobile-Agent-v2."""

  adb_path: str
  console_port: int
  api_url: str = ''
  token: str = ''
  action_model: str = 'gpt-4o'
  planning_model: str = 'gpt-4-turbo'
  caption_call_method: str = 'api'
  caption_model: str = ''
  add_info: str = (
      'If you want to tap an icon of an app, use the action "Open app". If you'
      ' want to exit an app, use the action "Home"'
  )
  reflection_switch: bool = True
  memory_switch: bool = True
  work_dir: str | None = None
  post_action_sleep: float = 5.0

  @property
  def adb_command_prefix(self) -> str:
    return f'{self.adb_path} -s emulator-{self.console_port}'

  @classmethod
  def from_env(cls, adb_path: str, console_port: int) -> 'MobileAgentV2Config':
    caption_call_method = os.environ.get(
        'MOBILE_AGENT_V2_CAPTION_CALL_METHOD', 'api'
    )
    return cls(
        adb_path=adb_path,
        console_port=console_port,
        api_url=os.environ.get('MOBILE_AGENT_V2_API_URL', ''),
        token=os.environ.get('MOBILE_AGENT_V2_API_TOKEN', ''),
        action_model=os.environ.get(
            'MOBILE_AGENT_V2_ACTION_MODEL',
            os.environ.get('MOBILE_AGENT_V2_MODEL', 'gpt-4o'),
        ),
        planning_model=os.environ.get(
            'MOBILE_AGENT_V2_PLANNING_MODEL',
            os.environ.get('MOBILE_AGENT_V2_MODEL', 'gpt-4-turbo'),
        ),
        caption_call_method=caption_call_method,
        caption_model=os.environ.get(
            'MOBILE_AGENT_V2_CAPTION_MODEL',
            os.environ.get('MOBILE_AGENT_V2_MODEL', ''),
        ),
        add_info=os.environ.get(
            'MOBILE_AGENT_V2_ADD_INFO',
            cls.__dataclass_fields__['add_info'].default,
        ),
        reflection_switch=_env_bool('MOBILE_AGENT_V2_REFLECTION', True),
        memory_switch=_env_bool('MOBILE_AGENT_V2_MEMORY', True),
        work_dir=os.environ.get('MOBILE_AGENT_V2_WORK_DIR') or None,
        post_action_sleep=float(
            os.environ.get('MOBILE_AGENT_V2_POST_ACTION_SLEEP', '5.0')
        ),
    )

  def validate(self) -> None:
    if not self.api_url or not self.token:
      raise ValueError(
          'Mobile-Agent-v2 requires MOBILE_AGENT_V2_API_URL and '
          'MOBILE_AGENT_V2_API_TOKEN in the environment.'
      )
    if self.caption_call_method not in ('api', 'local'):
      raise ValueError(
          'MOBILE_AGENT_V2_CAPTION_CALL_METHOD must be "api" or "local".'
      )


@dataclasses.dataclass
class StepData:
  """Data emitted by one Mobile-Agent-v2 step."""

  done: bool
  data: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    return self.data


@contextlib.contextmanager
def _pushd(path: str):
  current = os.getcwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(current)


class MobileAgentV2Runner:
  """Single-step runner extracted from Mobile-Agent-v2's original loop."""

  def __init__(self, config: MobileAgentV2Config):
    self._config = config
    self._work_dir = config.work_dir or tempfile.mkdtemp(
        prefix='mobile_agent_v2_'
    )
    self._temp_dir = os.path.join(self._work_dir, 'temp')
    self._screenshot_dir = os.path.join(self._work_dir, 'screenshot')
    self._loaded = False
    self._initialized = False

  def reset(self) -> None:
    self._initialized = False
    self._thought_history = []
    self._summary_history = []
    self._action_history = []
    self._summary = ''
    self._action = ''
    self._completed_requirements = ''
    self._memory = ''
    self._insight = ''
    self._error_flag = False
    self._keyboard = False
    self._keyboard_height_limit = 0
    self._perception_infos = []
    self._width = 0
    self._height = 0
    self._prepare_work_dirs()

  def _prepare_work_dirs(self) -> None:
    os.makedirs(self._work_dir, exist_ok=True)
    for path in (self._temp_dir, self._screenshot_dir):
      if os.path.exists(path):
        shutil.rmtree(path)
      os.makedirs(path)

  def _load_dependencies(self) -> None:
    if self._loaded:
      return
    if not os.path.isdir(_MOBILE_AGENT_V2_ROOT):
      raise FileNotFoundError(
          f'Mobile-Agent-v2 root not found: {_MOBILE_AGENT_V2_ROOT}'
      )
    if _MOBILE_AGENT_V2_ROOT not in sys.path:
      sys.path.insert(0, _MOBILE_AGENT_V2_ROOT)

    _install_tf_keras_legacy_layers_shim()

    self._api = importlib.import_module('MobileAgent.api')
    self._text_localization = importlib.import_module(
        'MobileAgent.text_localization'
    )
    self._icon_localization = importlib.import_module(
        'MobileAgent.icon_localization'
    )
    self._controller = importlib.import_module('MobileAgent.controller')
    self._prompt = importlib.import_module('MobileAgent.prompt')
    self._chat = importlib.import_module('MobileAgent.chat')

    modelscope_pipelines = importlib.import_module('modelscope.pipelines')
    modelscope_tasks = importlib.import_module('modelscope.utils.constant')
    modelscope = importlib.import_module('modelscope')

    torch = importlib.import_module('torch')

    torch.manual_seed(1234)
    self._caption_tokenizer = None
    self._caption_model = None
    if self._config.caption_call_method == 'local':
      if self._config.caption_model == 'qwen-vl-chat':
        model_dir = modelscope.snapshot_download(
            'qwen/Qwen-VL-Chat', revision='v1.1.0'
        )
        auto_model = modelscope.AutoModelForCausalLM
        auto_tokenizer = modelscope.AutoTokenizer
        generation_config = modelscope.GenerationConfig
        self._caption_model = auto_model.from_pretrained(
            model_dir, device_map='cuda', trust_remote_code=True
        ).eval()
        self._caption_model.generation_config = (
            generation_config.from_pretrained(model_dir, trust_remote_code=True)
        )
        self._caption_tokenizer = auto_tokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
      elif self._config.caption_model == 'qwen-vl-chat-int4':
        model_dir = modelscope.snapshot_download(
            'qwen/Qwen-VL-Chat-Int4', revision='v1.0.0'
        )
        auto_model = modelscope.AutoModelForCausalLM
        auto_tokenizer = modelscope.AutoTokenizer
        generation_config = modelscope.GenerationConfig
        self._caption_model = auto_model.from_pretrained(
            model_dir,
            device_map='cuda',
            trust_remote_code=True,
            use_safetensors=True,
        ).eval()
        self._caption_model.generation_config = (
            generation_config.from_pretrained(
                model_dir, trust_remote_code=True, do_sample=False
            )
        )
        self._caption_tokenizer = auto_tokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
      else:
        raise ValueError(
            'Local caption model must be qwen-vl-chat or qwen-vl-chat-int4.'
        )
    elif self._config.caption_call_method != 'api':
      raise ValueError('caption_call_method must be "api" or "local".')

    groundingdino_dir = modelscope.snapshot_download(
        'AI-ModelScope/GroundingDINO', revision='v1.0.0'
    )
    self._groundingdino_model = modelscope_pipelines.pipeline(
        'grounding-dino-task', model=groundingdino_dir
    )
    self._ocr_detection = modelscope_pipelines.pipeline(
        modelscope_tasks.Tasks.ocr_detection,
        model='damo/cv_resnet18_ocr-detection-line-level_damo',
    )
    self._ocr_recognition = modelscope_pipelines.pipeline(
        modelscope_tasks.Tasks.ocr_recognition,
        model='damo/cv_convnextTiny_ocr-recognition-document_damo',
    )
    self._loaded = True

  def _get_all_files_in_folder(self, folder_path: str) -> list[str]:
    return os.listdir(folder_path)

  def _draw_coordinates_on_image(
      self, image_path: str, coordinates: list[list[float]]
  ) -> str:
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
      draw.ellipse(
          (
              coord[0] - point_size,
              coord[1] - point_size,
              coord[0] + point_size,
              coord[1] + point_size,
          ),
          fill='red',
      )
    output_image_path = os.path.join(self._screenshot_dir, 'output_image.png')
    image.save(output_image_path)
    return output_image_path

  def _crop(self, image_path: str, box: list[float], index: int) -> None:
    image = Image.open(image_path)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2 - 10 or y1 >= y2 - 10:
      return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(os.path.join(self._temp_dir, f'{index}.jpg'))

  def _generate_local(self, image_file: str, query: str) -> str:
    query = self._caption_tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = self._caption_model.chat(
        self._caption_tokenizer, query=query, history=None
    )
    return response

  def _process_image(self, image: str, query: str) -> str:
    try:
      chat = self._chat.init_memory_chat()
      chat = self._chat.add_response('user', query, chat, image)
      return self._api.inference_chat(
          chat,
          self._config.caption_model or self._config.action_model,
          self._config.api_url,
          self._config.token,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('Mobile-Agent-v2 icon caption failed.')
      return 'This is an icon.'

  def _generate_api(self, images: list[str], query: str) -> dict[int, str]:
    concurrent_futures = importlib.import_module('concurrent.futures')
    icon_map = {}
    with concurrent_futures.ThreadPoolExecutor() as executor:
      futures = {
          executor.submit(self._process_image, image, query): i
          for i, image in enumerate(images)
      }
      for future in concurrent_futures.as_completed(futures):
        i = futures[future]
        icon_map[i + 1] = future.result()
    return icon_map

  def _merge_text_blocks(
      self, text_list: list[str], coordinates_list: list[list[float]]
  ) -> tuple[list[str], list[list[float]]]:
    merged_text_blocks = []
    merged_coordinates = []

    sorted_indices = sorted(
        range(len(coordinates_list)),
        key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]),
    )
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks

    for i in range(num_blocks):
      if merge[i]:
        continue
      anchor = i
      group_text = [sorted_text_list[anchor]]
      group_coordinates = [sorted_coordinates_list[anchor]]

      for j in range(i + 1, num_blocks):
        if merge[j]:
          continue
        same_x = abs(sorted_coordinates_list[anchor][0]
                     - sorted_coordinates_list[j][0]) < 10
        vertical_gap = (
            sorted_coordinates_list[j][1]
            - sorted_coordinates_list[anchor][3]
        )
        similar_height = abs(
            sorted_coordinates_list[anchor][3]
            - sorted_coordinates_list[anchor][1]
            - (
                sorted_coordinates_list[j][3]
                - sorted_coordinates_list[j][1]
            )
        ) < 10
        if same_x and -10 <= vertical_gap < 30 and similar_height:
          group_text.append(sorted_text_list[j])
          group_coordinates.append(sorted_coordinates_list[j])
          merge[anchor] = True
          anchor = j
          merge[anchor] = True

      merged_text = '\n'.join(group_text)
      min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
      min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
      max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
      max_y2 = max(group_coordinates, key=lambda x: x[3])[3]
      merged_text_blocks.append(merged_text)
      merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])

    return merged_text_blocks, merged_coordinates

  def _get_perception_infos(
      self, screenshot_file: str
  ) -> tuple[list[dict[str, Any]], int, int]:
    self._controller.get_screenshot(self._config.adb_command_prefix)
    width, height = Image.open(screenshot_file).size

    text, coordinates = self._text_localization.ocr(
        screenshot_file, self._ocr_detection, self._ocr_recognition
    )
    text, coordinates = self._merge_text_blocks(text, coordinates)

    center_list = [
        [(coordinate[0] + coordinate[2]) / 2,
         (coordinate[1] + coordinate[3]) / 2]
        for coordinate in coordinates
    ]
    self._draw_coordinates_on_image(screenshot_file, center_list)

    perception_infos = []
    for i in range(len(coordinates)):
      perception_infos.append({
          'text': 'text: ' + text[i],
          'coordinates': coordinates[i],
      })

    coordinates = self._icon_localization.det(
        screenshot_file, 'icon', self._groundingdino_model
    )
    for coordinate in coordinates:
      perception_infos.append({'text': 'icon', 'coordinates': coordinate})

    image_box = []
    image_id = []
    for i, perception_info in enumerate(perception_infos):
      if perception_info['text'] == 'icon':
        image_box.append(perception_info['coordinates'])
        image_id.append(i)

    for i, box in enumerate(image_box):
      self._crop(screenshot_file, box, image_id[i])

    images = self._get_all_files_in_folder(self._temp_dir)
    if images:
      images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
      image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
      prompt = (
          'This image is an icon from a phone screen. Please briefly describe'
          ' the shape and color of this icon in one sentence.'
      )
      if self._config.caption_call_method == 'local':
        icon_map = {}
        for i, image in enumerate(images):
          image_path = os.path.join(self._temp_dir, image)
          icon_width, icon_height = Image.open(image_path).size
          if icon_height > 0.8 * height or (
              icon_width * icon_height > 0.2 * width * height
          ):
            des = 'None'
          else:
            des = self._generate_local(image_path, prompt)
          icon_map[i + 1] = des
      else:
        icon_map = self._generate_api(
            [os.path.join(self._temp_dir, image) for image in images], prompt
        )
      for i, j in zip(image_id, range(1, len(image_id) + 1)):
        if icon_map.get(j):
          perception_infos[i]['text'] = 'icon: ' + icon_map[j]

    for perception_info in perception_infos:
      coordinate = perception_info['coordinates']
      perception_info['coordinates'] = [
          int((coordinate[0] + coordinate[2]) / 2),
          int((coordinate[1] + coordinate[3]) / 2),
      ]

    return perception_infos, width, height

  def _refresh_keyboard_state(self) -> None:
    self._keyboard = False
    for perception_info in self._perception_infos:
      if perception_info['coordinates'][1] < self._keyboard_height_limit:
        continue
      if 'ADB Keyboard' in perception_info['text']:
        self._keyboard = True
        break

  def _initialize_if_needed(self) -> None:
    self._load_dependencies()
    if self._initialized:
      return
    self.reset()
    screenshot_file = os.path.join(self._screenshot_dir, 'screenshot.jpg')
    with _pushd(self._work_dir):
      self._perception_infos, self._width, self._height = (
          self._get_perception_infos(screenshot_file)
      )
    shutil.rmtree(self._temp_dir)
    os.makedirs(self._temp_dir)
    self._keyboard_height_limit = 0.9 * self._height
    self._refresh_keyboard_state()
    self._initialized = True

  def run_one_step(self, goal: str) -> StepData:
    self._initialize_if_needed()
    screenshot_file = os.path.join(self._screenshot_dir, 'screenshot.jpg')
    last_screenshot_file = os.path.join(
        self._screenshot_dir, 'last_screenshot.jpg'
    )
    data: dict[str, Any] = {
        'screenshot_path': screenshot_file,
        'goal': goal,
    }

    try:
      prompt_action = self._prompt.get_action_prompt(
          goal,
          self._perception_infos,
          self._width,
          self._height,
          self._keyboard,
          self._summary_history,
          self._action_history,
          self._summary,
          self._action,
          self._config.add_info,
          self._error_flag,
          self._completed_requirements,
          self._memory,
      )
      chat_action = self._chat.init_action_chat()
      chat_action = self._chat.add_response(
          'user', prompt_action, chat_action, screenshot_file
      )
      output_action = self._api.inference_chat(
          chat_action,
          self._config.action_model,
          self._config.api_url,
          self._config.token,
      )
      thought = (
          output_action.split('### Thought ###')[-1]
          .split('### Action ###')[0]
          .replace('\n', ' ')
          .replace(':', '')
          .replace('  ', ' ')
          .strip()
      )
      self._summary = (
          output_action.split('### Operation ###')[-1]
          .replace('\n', ' ')
          .replace('  ', ' ')
          .strip()
      )
      self._action = (
          output_action.split('### Action ###')[-1]
          .split('### Operation ###')[0]
          .replace('\n', ' ')
          .replace('  ', ' ')
          .strip()
      )
      chat_action = self._chat.add_response(
          'assistant', output_action, chat_action
      )
      data.update({
          'v2_prompt': prompt_action,
          'v2_raw_response': output_action,
          'v2_thought': thought,
          'v2_action': self._action,
          'v2_operation_summary': self._summary,
      })

      if self._config.memory_switch:
        prompt_memory = self._prompt.get_memory_prompt(self._insight)
        chat_action = self._chat.add_response(
            'user', prompt_memory, chat_action
        )
        output_memory = self._api.inference_chat(
            chat_action,
            self._config.action_model,
            self._config.api_url,
            self._config.token,
        )
        chat_action = self._chat.add_response(
            'assistant', output_memory, chat_action
        )
        data['memory_output'] = output_memory
        output_memory = (
            output_memory.split('### Important content ###')[-1]
            .split('\n\n')[0]
            .strip()
            + '\n'
        )
        if 'None' not in output_memory and output_memory not in self._memory:
          self._memory += output_memory

      execution_result = self._execute_action(screenshot_file)
      data['execution_result'] = execution_result
      if execution_result == 'stop':
        return StepData(done=True, data=data)

      time.sleep(self._config.post_action_sleep)
      self._update_after_action(
          goal, thought, screenshot_file, last_screenshot_file, data
      )
      return StepData(done=False, data=data)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.exception('Mobile-Agent-v2 step failed.')
      data['execution_error'] = repr(exc)
      return StepData(done=False, data=data)

  def _execute_action(self, screenshot_file: str) -> str:
    action = self._action
    adb_path = self._config.adb_command_prefix
    if 'Open app' in action:
      app_name = action.split('(')[-1].split(')')[0]
      text, coordinate = self._text_localization.ocr(
          screenshot_file, self._ocr_detection, self._ocr_recognition
      )
      for ti in range(len(text)):
        if app_name == text[ti]:
          name_coordinate = [
              int((coordinate[ti][0] + coordinate[ti][2]) / 2),
              int((coordinate[ti][1] + coordinate[ti][3]) / 2),
          ]
          self._controller.tap(
              adb_path,
              name_coordinate[0],
              name_coordinate[1] - int(coordinate[ti][3] - coordinate[ti][1]),
          )
          return f'open_app:{app_name}'
      return f'open_app_not_found:{app_name}'

    if 'Tap' in action:
      coordinate = action.split('(')[-1].split(')')[0].split(', ')
      x, y = int(coordinate[0]), int(coordinate[1])
      self._controller.tap(adb_path, x, y)
      return f'tap:{x},{y}'

    if 'Swipe' in action:
      coordinate1 = action.split('Swipe (')[-1].split('), (')[0].split(', ')
      coordinate2 = action.split('), (')[-1].split(')')[0].split(', ')
      x1, y1 = int(coordinate1[0]), int(coordinate1[1])
      x2, y2 = int(coordinate2[0]), int(coordinate2[1])
      self._controller.slide(adb_path, x1, y1, x2, y2)
      return f'swipe:{x1},{y1}->{x2},{y2}'

    if action.startswith('Type'):
      text = self._parse_type_text(action)
      self._controller.type(adb_path, text)
      return 'type'

    if 'Back' in action:
      self._controller.back(adb_path)
      return 'back'

    if 'Home' in action:
      self._controller.home(adb_path)
      return 'home'

    if 'Stop' in action:
      return 'stop'

    return 'unknown_action'

  def _parse_type_text(self, action: str) -> str:
    """Extracts text from Mobile-Agent-v2 Type actions."""
    patterns = (
        r'^Type\s*\((?P<text>.*)\)\s*$',
        r'^Type\s+"(?P<text>.*)"\s*$',
        r"^Type\s+'(?P<text>.*)'\s*$",
        r'^Type\s*:\s*Type\s+"(?P<text>.*)"\s*$',
        r"^Type\s*:\s*Type\s+'(?P<text>.*)'\s*$",
        r'^Type\s*\(text\)\s*:\s*Type\s+"(?P<text>.*)"\s*$',
        r"^Type\s*\(text\)\s*:\s*Type\s+'(?P<text>.*)'\s*$",
    )
    for pattern in patterns:
      match = re.fullmatch(pattern, action)
      if match:
        return match.group('text')
    raise ValueError(f'Invalid Type action format: {action!r}')

  def _update_after_action(
      self,
      goal: str,
      thought: str,
      screenshot_file: str,
      last_screenshot_file: str,
      data: dict[str, Any],
  ) -> None:
    last_perception_infos = copy.deepcopy(self._perception_infos)
    last_keyboard = self._keyboard
    if os.path.exists(last_screenshot_file):
      os.remove(last_screenshot_file)
    os.rename(screenshot_file, last_screenshot_file)

    with _pushd(self._work_dir):
      self._perception_infos, self._width, self._height = (
          self._get_perception_infos(screenshot_file)
      )
    shutil.rmtree(self._temp_dir)
    os.makedirs(self._temp_dir)
    self._refresh_keyboard_state()

    if self._config.reflection_switch:
      prompt_reflect = self._prompt.get_reflect_prompt(
          goal,
          last_perception_infos,
          self._perception_infos,
          self._width,
          self._height,
          last_keyboard,
          self._keyboard,
          self._summary,
          self._action,
          self._config.add_info,
      )
      chat_reflect = self._chat.init_reflect_chat()
      chat_reflect = self._chat.add_response_two_image(
          'user',
          prompt_reflect,
          chat_reflect,
          [last_screenshot_file, screenshot_file],
      )
      output_reflect = self._api.inference_chat(
          chat_reflect,
          self._config.action_model,
          self._config.api_url,
          self._config.token,
      )
      chat_reflect = self._chat.add_response(
          'assistant', output_reflect, chat_reflect
      )
      data['reflection_output'] = output_reflect
      reflect = (
          output_reflect.split('### Answer ###')[-1]
          .replace('\n', ' ')
          .strip()
      )

      if 'A' in reflect:
        self._accept_action_and_update_plan(goal, thought, data)
        self._error_flag = False
      elif 'B' in reflect:
        self._error_flag = True
        self._controller.back(self._config.adb_command_prefix)
      elif 'C' in reflect:
        self._error_flag = True
    else:
      self._accept_action_and_update_plan(goal, thought, data)

    if os.path.exists(last_screenshot_file):
      os.remove(last_screenshot_file)

  def _accept_action_and_update_plan(
      self, goal: str, thought: str, data: dict[str, Any]
  ) -> None:
    self._thought_history.append(thought)
    self._summary_history.append(self._summary)
    self._action_history.append(self._action)
    prompt_planning = self._prompt.get_process_prompt(
        goal,
        self._thought_history,
        self._summary_history,
        self._action_history,
        self._completed_requirements,
        self._config.add_info,
    )
    chat_planning = self._chat.init_memory_chat()
    chat_planning = self._chat.add_response(
        'user', prompt_planning, chat_planning
    )
    output_planning = self._api.inference_chat(
        chat_planning,
        self._config.planning_model,
        self._config.api_url,
        self._config.token,
    )
    self._chat.add_response('assistant', output_planning, chat_planning)
    data['planning_output'] = output_planning
    self._completed_requirements = (
        output_planning.split('### Completed contents ###')[-1]
        .replace('\n', ' ')
        .strip()
    )


class MobileAgentV2Agent(base_agent.EnvironmentInteractingAgent):
  """AndroidWorld agent shell that delegates each step to Mobile-Agent-v2."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      config: MobileAgentV2Config,
      name: str = 'mobile_agent_v2',
  ):
    super().__init__(env, name=name)
    config.validate()
    self._runner = MobileAgentV2Runner(config)

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home=go_home)
    self.env.hide_automation_ui()
    self._runner.reset()

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = self._runner.run_one_step(goal)
    return base_agent.AgentInteractionResult(
        done=step_data.done,
        data=step_data.to_dict(),
    )
