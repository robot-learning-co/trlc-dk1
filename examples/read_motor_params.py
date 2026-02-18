from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *
import serial
import time

motor=Motor(DM_Motor_Type.DM4340, 0x07, 0x17)

serial_device = serial.Serial('/dev/tty.usbmodem00000000050C1', 921600, timeout=0.5)
time.sleep(0.5)

control=MotorControl(serial_device)
control.addMotor(motor)

if control.read_motor_param(motor, DM_variable.CTRL_MODE) is not None:
    print("Motor is connected.")
else:
    raise Exception("Unable to read motor parameters.")


for param in DM_variable:
    value = control.read_motor_param(motor, param)
    print(f"{param.name:<9} : {value}")