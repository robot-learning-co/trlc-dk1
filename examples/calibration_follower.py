from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *

import serial
import time


follower_config = DK1FollowerConfig(
    port="/dev/ttyACM1",
)
follower = DK1Follower(follower_config)

follower.serial_device = serial.Serial(follower_config.port, 921600, timeout=0.5)
time.sleep(0.5)
follower.control = MotorControl(follower.serial_device)

for key, motor in follower.motors.items():
    follower.control.addMotor(motor)
    for _ in range(3):
        follower.control.refresh_motor_status(motor)
        time.sleep(0.01)
    
    if follower.control.read_motor_param(motor, DM_variable.CTRL_MODE) is not None:
        print(f"{key} ({motor.MotorType.name}) is connected.")
    else:
        raise Exception(f"Unable to read from {key} ({motor.MotorType.name}).")

for key, motor in follower.motors.items():
    follower.control.set_zero_position(motor)
    print(f"{key} ({motor.MotorType.name}) set to zero position.")
    
follower.control.serial_.close()