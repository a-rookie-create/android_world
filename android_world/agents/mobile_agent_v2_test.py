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

"""Tests for Mobile-Agent-v2 adapter."""

from absl.testing import absltest
from android_world.agents import mobile_agent_v2
from android_world.env import json_action


class MobileAgentV2AdapterTest(absltest.TestCase):

  def test_parse_v2_output(self):
    output = """### Thought ###
I need to open Contacts.
### Action ###
Tap (10, 20)
### Operation ###
Tap the Contacts icon."""

    thought, action, operation = mobile_agent_v2.parse_v2_output(output)

    self.assertEqual(thought, 'I need to open Contacts.')
    self.assertEqual(action, 'Tap (10, 20)')
    self.assertEqual(operation, 'Tap the Contacts icon.')

  def test_parse_tap_action(self):
    parsed = mobile_agent_v2.parse_v2_action('Tap (123, 456)')

    self.assertEqual(parsed.action_type, 'click')
    self.assertEqual(parsed.x, 123)
    self.assertEqual(parsed.y, 456)

  def test_parse_open_app_action_with_label(self):
    parsed = mobile_agent_v2.parse_v2_action('Open app (app name): Contacts')

    self.assertEqual(parsed.action_type, 'open_app')
    self.assertEqual(parsed.app_name, 'Contacts')

  def test_parse_swipe_action(self):
    parsed = mobile_agent_v2.parse_v2_action('Swipe (1, 2), (3, 4)')

    self.assertEqual(parsed.action_type, 'swipe')
    self.assertEqual(parsed.x, 1)
    self.assertEqual(parsed.y, 2)
    self.assertEqual(parsed.x2, 3)
    self.assertEqual(parsed.y2, 4)

  def test_parse_type_action(self):
    parsed = mobile_agent_v2.parse_v2_action('Type ("hello world")')

    self.assertEqual(parsed.action_type, 'input_text')
    self.assertEqual(parsed.text, 'hello world')

  def test_stop_converts_to_status_action(self):
    agent = mobile_agent_v2.MobileAgentV2(env=None)
    action = agent._to_json_action(mobile_agent_v2.parse_v2_action('Stop'))

    self.assertEqual(
        action,
        json_action.JSONAction(
            action_type=json_action.STATUS,
            goal_status='task_complete',
        ),
    )

  def test_swipe_is_not_represented_as_json_action(self):
    agent = mobile_agent_v2.MobileAgentV2(env=None)
    action = agent._to_json_action(
        mobile_agent_v2.parse_v2_action('Swipe (1, 2), (3, 4)')
    )

    self.assertIsNone(action)

  def test_type_is_not_represented_as_json_action(self):
    agent = mobile_agent_v2.MobileAgentV2(env=None)
    action = agent._to_json_action(
        mobile_agent_v2.parse_v2_action('Type ("hello")')
    )

    self.assertIsNone(action)


if __name__ == '__main__':
  absltest.main()
