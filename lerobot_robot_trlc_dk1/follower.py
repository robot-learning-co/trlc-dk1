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

from dataclasses import dataclass, field
from functools import cached_property
import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# POS_VEL mode constants
_DM4310_TORQUE_CONSTANT = 0.945    # Nm/A
_EMIT_VELOCITY_SCALE = 100         # rad/s multiplier
_EMIT_CURRENT_SCALE = 1000         # A multiplier
_DM4310_SPEED = 200 / 60 * 2 * np.pi   # 200 rpm → rad/s
_DM4340_SPEED = 52.5 / 60 * 2 * np.pi  # 52.5 rpm → rad/s
_JOINT_LIMITS = {
    "joint_4": (-100 / 180 * np.pi, 100 / 180 * np.pi),
    "joint_5": (-90 / 180 * np.pi, 90 / 180 * np.pi),
}


def _map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


@RobotConfig.register_subclass("dk1_follower")
@dataclass
class DK1FollowerConfig(RobotConfig):
    port: str
    # "impedance"    — MIT mode + gravity compensation (Python DK1Robot)
    # "rt_impedance" — MIT mode + gravity compensation (C++ RT control loop)
    # "pos_vel"      — original POS_VEL mode with velocity scaling
    control_mode: str = "rt_impedance"
    # Shared
    max_gripper_torque: float = 1.0         # Nm
    disable_torque_on_disconnect: bool = False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Impedance mode
    gravity_comp_scale: float = 1.0         # full gravity compensation
    # POS_VEL mode only
    joint_velocity_scaling: float = 0.2


class DK1Follower(Robot):
    """
    TRLC-DK1 Follower Arm — LeRobot wrapper.

    Two control modes selected via DK1FollowerConfig.control_mode:

      "impedance"  MIT mode + gravity compensation, 250 Hz background thread.
                   Wraps trlc_dk1_control.DK1Robot.  Non-blocking get/send.

      "pos_vel"    Original POS_VEL mode with velocity scaling.
                   Direct serial I/O on every get_observation() / send_action().
    """

    config_class = DK1FollowerConfig
    name = "dk1_follower"

    def __init__(self, config: DK1FollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        # Impedance mode state
        self._robot = None              # DK1Robot | None

        # POS_VEL mode state
        self._serial_device = None
        self._control = None
        self._motors = None             # dict[str, Motor] | None
        self._bus_connected = False
        self._gripper_open_pos = 0.0
        self._gripper_closed_pos = -4.7

    # ------------------------------------------------------------------
    # LeRobot feature descriptors  (same keys for both modes)
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        motor_ft = {f"{j}.pos": float for j in JOINT_NAMES}
        motor_ft["gripper.pos"] = float
        if self.config.control_mode in ("impedance", "rt_impedance"):
            for j in JOINT_NAMES:
                motor_ft[f"{j}.vel"] = float
                motor_ft[f"{j}.torque"] = float
            motor_ft["gripper.torque"] = float
        cam_ft = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
        return {**motor_ft, **cam_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in JOINT_NAMES} | {"gripper.pos": float}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        cams_ok = all(cam.is_connected for cam in self.cameras.values())
        if self.config.control_mode == "impedance":
            return (
                self._robot is not None
                and self._robot._motor_chain.is_running
                and cams_ok
            )
        elif self.config.control_mode == "rt_impedance":
            return self._robot is not None and cams_ok
        else:
            return self._bus_connected and cams_ok

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.config.control_mode == "impedance":
            from trlc_dk1_control import DK1Robot, DK1_DEFAULT_CONFIG
            cfg = DK1_DEFAULT_CONFIG(self.config.port)
            cfg.max_gripper_torque_nm = self.config.max_gripper_torque
            cfg.gravity_comp_scale = self.config.gravity_comp_scale
            self._robot = DK1Robot(cfg)
            self._robot.connect()
        elif self.config.control_mode == "rt_impedance":
            from trlc_dk1_control import DK1_DEFAULT_CONFIG
            from trlc_dk1_control.rt_robot import DK1RobotRT
            cfg = DK1_DEFAULT_CONFIG(self.config.port)
            cfg.max_gripper_torque_nm = self.config.max_gripper_torque
            cfg.gravity_comp_scale = self.config.gravity_comp_scale
            cfg.disable_torque_on_disconnect = self.config.disable_torque_on_disconnect
            self._robot = DK1RobotRT(cfg)
            self._robot.connect()
        else:
            self._connect_pos_vel()

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected (mode={self.config.control_mode}).")

    def _connect_pos_vel(self) -> None:
        import serial
        from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import (
            Motor, MotorControl, DM_Motor_Type,
        )

        self._serial_device = serial.Serial(self.config.port, 921600, timeout=0.5)
        time.sleep(0.5)
        self._control = MotorControl(self._serial_device)

        self._motors = {
            "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
            "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
            "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
            "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
            "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
            "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
            "gripper": Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
        }
        self._bus_connected = True
        self._configure_pos_vel()

    def _configure_pos_vel(self) -> None:
        from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import (
            Control_Type, DM_variable,
        )

        for key, motor in self._motors.items():
            self._control.addMotor(motor)
            for _ in range(3):
                self._control.refresh_motor_status(motor)
                time.sleep(0.01)
            if self._control.read_motor_param(motor, DM_variable.CTRL_MODE) is not None:
                print(f"{key} ({motor.MotorType.name}) is connected.")
                self._control.switchControlMode(motor, Control_Type.POS_VEL)
                self._control.enable(motor)
            else:
                raise RuntimeError(f"Unable to read from {key} ({motor.MotorType.name}).")

        for joint in ["joint_1", "joint_2", "joint_3"]:
            self._control.change_motor_param(self._motors[joint], DM_variable.ACC, 10.0)
            self._control.change_motor_param(self._motors[joint], DM_variable.DEC, -10.0)
            self._control.change_motor_param(self._motors[joint], DM_variable.KP_APR, 200)
            self._control.change_motor_param(self._motors[joint], DM_variable.KI_APR, 10)

        self._control.change_motor_param(self._motors["gripper"], DM_variable.KP_APR, 100)

        # Calibrate gripper
        self._control.switchControlMode(self._motors["gripper"], Control_Type.VEL)
        self._control.control_Vel(self._motors["gripper"], 10.0)
        while True:
            self._control.refresh_motor_status(self._motors["gripper"])
            if self._motors["gripper"].getTorque() > 1.2:
                self._control.control_Vel(self._motors["gripper"], 0.0)
                self._control.disable(self._motors["gripper"])
                self._control.set_zero_position(self._motors["gripper"])
                time.sleep(0.2)
                self._control.enable(self._motors["gripper"])
                break
            time.sleep(0.01)
        self._control.switchControlMode(self._motors["gripper"], Control_Type.Torque_Pos)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation / action
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode in ("impedance", "rt_impedance"):
            obs = self._get_observation_impedance()
        else:
            obs = self._get_observation_pos_vel()

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def _get_observation_impedance(self) -> dict[str, Any]:
        state = self._robot.get_joint_state()
        gripper = self._robot.get_gripper_state()
        obs = {}
        # Per-joint motor temperature (°C). The DAMIAO MIT reply carries T_MOS (data[6]) + T_ROTOR
        # (data[7]); the RT layer surfaces them as t_mos/t_rotor arrays. Report the HOTTER of the two
        # per joint so a monitor gates on the worst case (FR-016). When the RT layer doesn't provide
        # them the key is simply omitted — a consumer must read that as "unknown" (NaN), never 0 °C.
        t_mos, t_rotor = state.get("t_mos"), state.get("t_rotor")
        for i, j in enumerate(JOINT_NAMES):
            obs[f"{j}.pos"] = float(state["pos"][i])
            obs[f"{j}.vel"] = float(state["vel"][i])
            obs[f"{j}.torque"] = float(state["torque"][i])
            if t_mos is not None and t_rotor is not None:
                obs[f"{j}.temperature"] = float(max(t_mos[i], t_rotor[i]))
        obs["gripper.pos"] = gripper["pos"]
        obs["gripper.torque"] = gripper["torque"]
        if gripper.get("t_mos") is not None and gripper.get("t_rotor") is not None:
            obs["gripper.temperature"] = float(max(gripper["t_mos"], gripper["t_rotor"]))
        return obs

    def _get_observation_pos_vel(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        for key, motor in self._motors.items():
            self._control.refresh_motor_status(motor)
            if key == "gripper":
                obs[f"{key}.pos"] = _map_range(
                    motor.getPosition(),
                    self._gripper_open_pos, self._gripper_closed_pos,
                    0.0, 1.0,
                )
            else:
                obs[f"{key}.pos"] = motor.getPosition()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode in ("impedance", "rt_impedance"):
            q_des = np.array([action[f"{j}.pos"] for j in JOINT_NAMES])
            self._robot.command_joint_pos(q_des)
            self._robot.command_gripper(float(action["gripper.pos"]))
            return action
        else:
            return self._send_action_pos_vel(action)

    def _send_action_pos_vel(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }
        vel_scale = self.config.joint_velocity_scaling

        for key, motor in self._motors.items():
            if key == "gripper":
                self._control.refresh_motor_status(motor)
                gripper_goal = _map_range(
                    goal_pos[key], 0.0, 1.0,
                    self._gripper_open_pos, self._gripper_closed_pos,
                )
                self._control.control_pos_force(
                    motor, gripper_goal,
                    _DM4310_SPEED * _EMIT_VELOCITY_SCALE,
                    i_des=self.config.max_gripper_torque / _DM4310_TORQUE_CONSTANT * _EMIT_CURRENT_SCALE,
                )
            else:
                if key in _JOINT_LIMITS:
                    goal_pos[key] = float(np.clip(goal_pos[key], *_JOINT_LIMITS[key]))
                speed = _DM4310_SPEED if key in ("joint_4", "joint_5", "joint_6") else _DM4340_SPEED
                self._control.control_Pos_Vel(motor, goal_pos[key], vel_scale * speed)

        return {f"{k}.pos": v for k, v in goal_pos.items()}

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode in ("impedance", "rt_impedance"):
            self._robot.disconnect()
            self._robot = None
        else:
            if self.config.disable_torque_on_disconnect:
                for motor in self._motors.values():
                    self._control.disable(motor)
            else:
                self._serial_device.close()
            self._bus_connected = False
            self._motors = None
            self._control = None
            self._serial_device = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
