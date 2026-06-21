"""
DK1MotorChain — 250 Hz background thread for motor control.

Architecture:
  - One background thread (motor_thread_hz) handles all serial I/O.
  - The thread sends MIT commands to arm joints and EMIT commands to the gripper,
    then reads back motor state.  State and commands are shared with the server thread
    via a single threading.Lock.
  - Serial timeout is set to 5 ms (not 500 ms) so recv() never blocks the control loop.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

import numpy as np
import serial

# Add the DM_Control_Python directory to path so we can import DM_CAN
import os
_DM_CAN_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "lerobot_robot_trlc_dk1", "motors", "DM_Control_Python"
)
sys.path.insert(0, os.path.abspath(_DM_CAN_DIR))

from DM_CAN import (  # noqa: E402
    Motor,
    MotorControl,
    DM_Motor_Type,
    Control_Type,
    DM_variable,
)

from .config import DK1RobotConfig, DM4310_DQ_MAX

logger = logging.getLogger(__name__)

# Number of arm joints (excludes gripper)
NUM_ARM_JOINTS = 6


class DK1MotorChain:
    """
    Wraps DM_CAN.MotorControl and runs a 250 Hz background thread.

    Thread-safety: all shared state is protected by `_lock`.
    """

    def __init__(self, config: DK1RobotConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

        # Shared state (written by motor thread, read by server thread)
        self._pos = np.zeros(7)     # radians, [joint_1..joint_6, gripper]
        self._vel = np.zeros(7)     # rad/s
        self._torque = np.zeros(7)  # Nm

        # Shared commands (written by server thread, read by motor thread)
        self._arm_kp = config.arm_kp.copy()
        self._arm_kd = config.arm_kd.copy()
        self._arm_q_des = np.zeros(NUM_ARM_JOINTS)
        self._arm_dq_des = np.zeros(NUM_ARM_JOINTS)
        self._arm_tau_ff = np.zeros(NUM_ARM_JOINTS)

        self._gripper_q_des = config.gripper_open_pos
        self._gripper_vel = DM4310_DQ_MAX * config.EMIT_VELOCITY_SCALE
        self._gripper_i_des = (
            config.max_gripper_torque_nm
            / config.DM4310_TORQUE_CONSTANT
            * config.EMIT_CURRENT_SCALE
        )

        # Set by motor thread to indicate actual gripper zero after calibration
        self.gripper_open_pos: float = config.gripper_open_pos

        # Motor objects — created in start()
        self._motors: dict[str, Motor] = {}
        self._control: MotorControl | None = None
        self._serial_device: serial.Serial | None = None

        self._running = False
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None

        # Performance tracking
        self._loop_count = 0
        self._last_perf_log = time.monotonic()

    # -------------------------------------------------------------------------
    # Public API (thread-safe)
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Open serial port, configure motors, start background thread."""
        if self._running:
            raise RuntimeError("DK1MotorChain already running")

        self._serial_device = serial.Serial(
            self._config.serial_port,
            921600,
            timeout=self._config.serial_timeout,
        )
        time.sleep(0.5)

        self._control = MotorControl(self._serial_device)
        self._configure()

        # Initialise desired position to current position so we don't jump on start
        with self._lock:
            self._arm_q_des = self._pos[:NUM_ARM_JOINTS].copy()

        self._running = True
        self._thread = threading.Thread(target=self._motor_loop, daemon=True, name="dk1-motor")
        self._thread.start()
        logger.info("DK1MotorChain started at %.0f Hz", self._config.motor_thread_hz)

    def stop(self) -> None:
        """Stop the background thread and disable all motors."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._control is not None:
            for motor in self._motors.values():
                try:
                    self._control.disable(motor)
                except Exception:
                    pass
        if self._serial_device is not None:
            self._serial_device.close()
        logger.info("DK1MotorChain stopped")

    def set_arm_commands(
        self,
        kp: np.ndarray,
        kd: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        tau_ff: np.ndarray,
    ) -> None:
        """Update MIT command buffer for arm joints 1-6 (non-blocking)."""
        with self._lock:
            self._arm_kp = kp
            self._arm_kd = kd
            self._arm_q_des = q_des
            self._arm_dq_des = dq_des
            self._arm_tau_ff = tau_ff

    def set_gripper_command(self, q_des: float, vel: float, i_des: float) -> None:
        """Update EMIT command for gripper (non-blocking)."""
        with self._lock:
            self._gripper_q_des = q_des
            self._gripper_vel = vel
            self._gripper_i_des = i_des

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pos(7,), vel(7,), torque(7,)) — thread-safe copy."""
        with self._lock:
            return self._pos.copy(), self._vel.copy(), self._torque.copy()

    @property
    def is_running(self) -> bool:
        return self._running and self._thread_error is None

    def raise_if_failed(self) -> None:
        """Raise if the background motor thread has failed."""
        if self._thread_error is not None:
            raise RuntimeError("DK1MotorChain motor thread failed") from self._thread_error

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _configure(self) -> None:
        """Mirror follower.py configure(), but use MIT mode for arm joints."""
        assert self._control is not None

        arm_joint_names = [f"joint_{i}" for i in range(1, 7)]
        self._motors = {
            "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
            "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
            "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
            "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
            "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
            "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
            "gripper": Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
        }

        for key, motor in self._motors.items():
            self._control.addMotor(motor)
            for _ in range(3):
                self._control.refresh_motor_status(motor)
                time.sleep(0.01)
            if self._control.read_motor_param(motor, DM_variable.CTRL_MODE) is None:
                raise RuntimeError(f"Cannot communicate with motor {key!r}")
            logger.debug("%s (%s) detected", key, motor.MotorType.name)

        # Switch arm joints to MIT mode (motor turns green once enabled)
        for name in arm_joint_names:
            motor = self._motors[name]
            self._control.switchControlMode(motor, Control_Type.MIT)
            self._control.enable(motor)
            logger.info("%s (%s) connected", name, motor.MotorType.name)

        # Read initial arm state
        for i, name in enumerate(arm_joint_names):
            motor = self._motors[name]
            self._control.refresh_motor_status(motor)
            self._pos[i] = motor.getPosition()
            self._vel[i] = motor.getVelocity()
            self._torque[i] = motor.getTorque()

        # Calibrate gripper: open until torque threshold, set as zero
        self._calibrate_gripper()

    def _calibrate_gripper(self) -> None:
        """Spin gripper open until torque spike, then set zero position."""
        assert self._control is not None
        gripper = self._motors["gripper"]

        self._control.switchControlMode(gripper, Control_Type.VEL)
        self._control.enable(gripper)
        logger.info("gripper (%s) connected", gripper.MotorType.name)
        self._control.control_Vel(gripper, 10.0)

        while True:
            self._control.refresh_motor_status(gripper)
            if gripper.getTorque() > 0.7:
                self._control.control_Vel(gripper, 0.0)
                self._control.disable(gripper)
                self._control.set_zero_position(gripper)
                time.sleep(0.2)
                self._control.enable(gripper)
                break
            time.sleep(0.01)

        self.gripper_open_pos = gripper.getPosition()

        # Switch to EMIT (Torque_Pos) mode for force-controlled grasping
        self._control.switchControlMode(gripper, Control_Type.Torque_Pos)

        # Read initial gripper state
        self._control.refresh_motor_status(gripper)
        self._pos[6] = gripper.getPosition()
        self._vel[6] = gripper.getVelocity()
        self._torque[6] = gripper.getTorque()

        logger.info("Gripper calibrated: open position = %s", self.gripper_open_pos)

    # -------------------------------------------------------------------------
    # 250 Hz motor loop
    # -------------------------------------------------------------------------

    def _motor_loop(self) -> None:
        assert self._control is not None
        period = 1.0 / self._config.motor_thread_hz
        arm_names = [f"joint_{i}" for i in range(1, 7)]

        try:
            while self._running:
                t_start = time.monotonic()

                # Snapshot commands under lock
                with self._lock:
                    kp = self._arm_kp.copy()
                    kd = self._arm_kd.copy()
                    q_des = self._arm_q_des.copy()
                    dq_des = self._arm_dq_des.copy()
                    tau_ff = self._arm_tau_ff.copy()
                    g_q_des = self._gripper_q_des
                    g_vel = self._gripper_vel
                    g_i_des = self._gripper_i_des

                # Send MIT commands to arm joints and read feedback
                for i, name in enumerate(arm_names):
                    motor = self._motors[name]
                    self._control.controlMIT(
                        motor,
                        float(kp[i]),
                        float(kd[i]),
                        float(q_des[i]),
                        float(dq_des[i]),
                        float(tau_ff[i]),
                    )

                # Send EMIT command to gripper
                self._control.control_pos_force(
                    self._motors["gripper"],
                    float(g_q_des),
                    float(g_vel),
                    float(g_i_des),
                )

                # Collect motor state (updated by recv() inside each control call)
                new_pos = np.empty(7)
                new_vel = np.empty(7)
                new_torque = np.empty(7)
                for i, name in enumerate(arm_names):
                    m = self._motors[name]
                    new_pos[i] = m.getPosition()
                    new_vel[i] = m.getVelocity()
                    new_torque[i] = m.getTorque()
                gm = self._motors["gripper"]
                new_pos[6] = gm.getPosition()
                new_vel[6] = gm.getVelocity()
                new_torque[6] = gm.getTorque()

                # Update shared state buffer
                with self._lock:
                    self._pos = new_pos
                    self._vel = new_vel
                    self._torque = new_torque

                # Maintain loop period — sleep most of the time, busywait the tail
                # for precision (time.sleep has ~1-2 ms granularity on macOS)
                elapsed = time.monotonic() - t_start
                sleep_time = period - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time - 0.001)
                while time.monotonic() - t_start < period:
                    pass

                # Periodic performance print
                self._loop_count += 1
                now = time.monotonic()
                if now - self._last_perf_log >= 5.0:
                    hz = self._loop_count / (now - self._last_perf_log)
                    logger.debug(
                        "[motor]  %6.1f Hz  (target %.0f Hz)  loop=%.2f ms",
                        hz,
                        self._config.motor_thread_hz,
                        elapsed * 1e3,
                    )
                    self._loop_count = 0
                    self._last_perf_log = now
        except BaseException as exc:
            self._thread_error = exc
            self._running = False
            logger.exception("DK1MotorChain motor thread failed")
