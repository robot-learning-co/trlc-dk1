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
        self.gripper_closed_pos: float = config.gripper_closed_pos

        # Motor objects — created in start()
        self._motors: dict[str, Motor] = {}
        self._control: MotorControl | None = None
        self._serial_device: serial.Serial | None = None

        self._running = False
        self._thread: threading.Thread | None = None

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
        return self._running

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
            print(f"{key} ({motor.MotorType.name}) connected")

        # Switch arm joints to MIT mode
        for name in arm_joint_names:
            motor = self._motors[name]
            self._control.switchControlMode(motor, Control_Type.MIT)
            self._control.enable(motor)

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
        """Spin gripper open until stall, set zero, then spin closed until stall to find range."""
        assert self._control is not None
        gripper = self._motors["gripper"]

        # Use MIT mode with small torque and damping for safety
        self._control.switchControlMode(gripper, Control_Type.MIT)
        self._control.enable(gripper)

        # Calibration parameters
        tau_cal_open = 0.8  # Nm
        tau_cal_close = 0.8  # Nm
        kd_cal = 0.1   # rad/s damping
        stall_threshold_dq = 0.2  # rad/s
        stall_duration = 0.2  # seconds
        
        def move_until_stall(torque: float, label: str):
            print(f"Calibrating gripper: moving to {label} end-stop with {torque:.2f} Nm torque...")
            stall_start_time = None
            while True:
                # Send MIT command: no position control, only constant torque + damping
                self._control.controlMIT(gripper, 0.0, kd_cal, 0.0, 0.0, torque)
                
                dq = gripper.getVelocity()
                if abs(dq) < stall_threshold_dq:
                    if stall_start_time is None:
                        stall_start_time = time.monotonic()
                    elif time.monotonic() - stall_start_time > stall_duration:
                        # Stalled at end-stop
                        break
                else:
                    stall_start_time = None
                
                time.sleep(0.01)

        # 1. Move to OPEN
        move_until_stall(tau_cal_open, "OPEN")

        # Stop torque
        self._control.controlMIT(gripper, 0.0, 0.1, 0.0, 0.0, 0.0)
        time.sleep(0.1)

        # Set zero position at OPEN
        self._control.disable(gripper)
        self._control.set_zero_position(gripper)
        time.sleep(0.2)
        self._control.enable(gripper)
        
        self.gripper_open_pos = 0.0
        self._config.gripper_open_pos = 0.0
        
        # 2. Move to CLOSED (negative torque)
        move_until_stall(-tau_cal_close, "CLOSED")
        
        # Record closed position
        self._control.refresh_motor_status(gripper)
        self.gripper_closed_pos = gripper.getPosition()
        self._config.gripper_closed_pos = self.gripper_closed_pos

        # Stop torque
        self._control.controlMIT(gripper, 0.0, 0.1, 0.0, 0.0, 0.0)
        time.sleep(0.1)

        # Switch to EMIT (Torque_Pos) mode for force-controlled grasping
        self._control.switchControlMode(gripper, Control_Type.Torque_Pos)
        
        # Read initial gripper state
        self._control.refresh_motor_status(gripper)
        self._pos[6] = gripper.getPosition()
        self._vel[6] = gripper.getVelocity()
        self._torque[6] = gripper.getTorque()

        print(f"Gripper calibrated: open={self._config.gripper_open_pos:.4f}, closed={self._config.gripper_closed_pos:.4f}")

    # -------------------------------------------------------------------------
    # 250 Hz motor loop
    # -------------------------------------------------------------------------

    def _motor_loop(self) -> None:
        assert self._control is not None
        period = 1.0 / self._config.motor_thread_hz
        arm_names = [f"joint_{i}" for i in range(1, 7)]

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
                print(f"[motor]  {hz:6.1f} Hz  (target {self._config.motor_thread_hz:.0f} Hz)  loop={elapsed*1e3:.2f} ms")
                self._loop_count = 0
                self._last_perf_log = now
