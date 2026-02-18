#   Copyright 2025 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from dataclasses import dataclass
import logging
from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("bi_dk1_leader")
@dataclass
class BiDK1LeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str
    gripper_open_pos: int = 2280
    gripper_closed_pos: int = 1670
    
    
class BiDK1Leader(Teleoperator):
    config_class = BiDK1LeaderConfig
    name = "bi_dk1_leader"

    def __init__(self, config: BiDK1LeaderConfig):
        super().__init__(config)
        self.config = config
        
        left_arm_config = DK1LeaderConfig(
            port=self.config.left_arm_port,
            gripper_open_pos=self.config.gripper_open_pos,
            gripper_closed_pos=self.config.gripper_closed_pos,
        )
        right_arm_config = DK1LeaderConfig(
            port=self.config.right_arm_port,
            gripper_open_pos=self.config.gripper_open_pos,
            gripper_closed_pos=self.config.gripper_closed_pos,
        )
        
        self.left_arm = DK1Leader(left_arm_config)
        self.right_arm = DK1Leader(right_arm_config)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = False) -> None:
        self.left_arm.connect()
        self.right_arm.connect()    

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()
        
    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})
        
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()