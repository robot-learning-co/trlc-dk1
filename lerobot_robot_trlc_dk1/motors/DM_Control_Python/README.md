# DaMiao Python Library

Language: [中文](README_zh.md) | [ENGLISH](README.md) 

This library supports **macOS**, **Linux**, and **Windows** platforms.

🎉 **Join the QQ group for DaMiao motor technical discussions**: [677900232].  
🛒 **Visit the DaMiao Store on Taobao**: [DaMiao Intelligent Control Store](https://shop290016675.taobao.com/?spm=pc_detail.29232929/evo365560b447259.shop_block.dshopinfo.59f47dd6w4Z4dX).

---

## Table of Contents
1. [Importing the DaMiao Library](#1-importing-the-DaMiao-library)
2. [Defining Motor Control Objects](#2-defining-motor-control-objects)
3. [Motor States](#3-motor-states)
   - [Adding Motors](#31-adding-motors)
   - [Enabling Motors](#32-enabling-motors)
   - [Setting Zero Position](#33-setting-zero-position)
   - [Disabling Motors](#34-disabling-motors)
   - [Fetching Motor Status](#35-fetching-motor-status)
4. [Motor Control Modes](#4-motor-control-modes)
   - [MIT Mode](#41-mit-mode)
   - [Position-Velocity Mode](#42-position-velocity-mode)
   - [Velocity Mode](#43-velocity-mode)
   - [Force-Position Mixed Mode](#44-force-position-mixed-mode)
5. [Reading Motor States](#5-reading-motor-states)
6. [Modifying Motor Parameters](#6-modifying-motor-parameters)
   - [Changing Control Mode](#61-changing-control-mode)
   - [Saving Parameters](#62-saving-parameters)
   - [Reading Internal Registers](#63-reading-internal-registers)
   - [Writing Internal Registers](#64-writing-internal-registers)

---

## 1. Importing the DaMiao Library

Ensure that `DM_CAN.py` is in your project folder. Import it as follows:

```python
from DM_CAN import *
```
The library depends on the following Python packages: `serial`, `numpy`. Install them using:

```bash
pip install -r requirements.txt
```

* * *

## 2. Defining Motor Control Objects

Create motor objects using:

```python
Motor1 = Motor(DM_Motor_Type.DM4310, 0x01, 0x11)
Motor2 = Motor(DM_Motor_Type.DM4310, 0x02, 0x12)
Motor3 = Motor(DM_Motor_Type.DM4310, 0x03, 0x13)
```

* **First Parameter**: Motor type
* **Second Parameter**: Slave ID (Motor's CAN ID)
* **Third Parameter**: Master ID (Host ID; must be unique and not `0x00`)

Set up the serial device (example for Windows):

```python
serial_device = serial.Serial('COM8', 921600, timeout=0.5)
```

Initialize the motor control object:

```python
MotorControl1 = MotorControl(serial_device)
```

* * *

## 3. Motor States

### 3.1 Adding Motors

```python
MotorControl1.addMotor(Motor1)
MotorControl1.addMotor(Motor2)
MotorControl1.addMotor(Motor3)
```

### 3.2 Enabling Motors

Enable motors (ensure parameters are set before enabling):

```python
MotorControl1.enable(Motor1)
MotorControl1.enable(Motor2)
MotorControl1.enable(Motor3)
```

For older firmware, specify the control mode:

```python
MotorControl1.enable_old(Motor1, Control_Type.MIT)
MotorControl1.enable_old(Motor2, Control_Type.POS_VEL)
MotorControl1.enable_old(Motor3, Control_Type.VEL)
```

### 3.3 Setting Zero Position

Move the motor to the desired zero position (in disabled state) and set it as the new zero position:

```python
MotorControl1.set_zero_position(Motor3)
MotorControl1.set_zero_position(Motor6)
```

### 3.4 Disabling Motors

```python
MotorControl1.disable(Motor3)
MotorControl1.disable(Motor6)
```

### 3.5 Fetching Motor Status

To fetch motor status (torque, position, velocity) without sending control commands:

```python
MotorControl1.refresh_motor_status(Motor1)
print("Motor1:","POS:", Motor1.getPosition(), "VEL:", Motor1.getVelocity(), "TORQUE:", Motor1.getTorque())
```

* * *

## 4. Motor Control Modes

### 4.1 MIT Mode

After enabling the motor, use MIT mode for control:

```python
MotorControl1.controlMIT(Motor1, 50, 0.3, 0, 0, 0)
```

### 4.2 Position-Velocity Mode

Control motor position and velocity:

```python
q = math.sin(time.time())
MotorControl1.control_Pos_Vel(Motor1, q*10, 2)
```

### 4.3 Velocity Mode

Control motor velocity:

```python
q = math.sin(time.time())
MotorControl1.control_Vel(Motor1, q*5)
```

### 4.4 Force-Position Mixed Mode

Control motor position and force:

```python
MotorControl1.control_pos_force(Motor1, 10, 1000, 100)
```

* * *

## 5. Reading Motor States

Motor states are stored in the motor object. You can read the following parameters:

* Velocity
* Position
* Torque

Use the following functions:

```python
vel = Motor1.getVelocity()    # Get motor velocity
pos = Motor1.getPosition()    # Get motor position
tau = Motor1.getTorque()      # Get motor torque
```

After refreshing motor status:

```python
MotorControl1.refresh_motor_status(Motor1)
print("Motor1:", "POS:", Motor1.getPosition(), "VEL:", Motor1.getVelocity(), "TORQUE:", Motor1.getTorque())
```

* * *

## 6. Modifying Motor Parameters

### 6.1 Changing Control Mode

You can change the motor's control mode using the following function:

```python
if MotorControl1.switchControlMode(Motor1, Control_Type.POS_VEL):
    print("Switch to POS_VEL mode success")
if MotorControl1.switchControlMode(Motor2, Control_Type.VEL):
    print("Switch to VEL mode success")
```

Note: The change is only temporary and will be reset after power cycle.

### 6.2 Saving Parameters

To save all changed parameters to the motor's flash memory:

```python
MotorControl1.save_motor_param(Motor1)
```

### 6.3 Reading Internal Registers

You can read internal registers of the motor, such as:

```python
print("PMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.PMAX))
print("MST_ID:", MotorControl1.read_motor_param(Motor1, DM_variable.MST_ID))
print("VMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.VMAX))
print("TMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.TMAX))
```

Use `getParam` to retrieve stored values:

```python
print("PMAX", Motor1.getParam(DM_variable.PMAX))
```

### 6.4 Writing Internal Registers

Some internal register values can be modified. Here's an example:

```python
if MotorControl1.change_motor_param(Motor1, DM_variable.KP_APR, 54):
   print("Write success")
```

Note: Changes will be lost after power cycle unless saved to flash memory.

### 7.Internal Register Parameter List **Note**: 

**Note**:  

- `RW`: Read and Write  
- `RO`: Read Only   - Corresponding variable names can be directly used in `DM_variable`.

| Register Address (Decimal) | Variable  | Description                 | R/W  | Range           | Data Type |
| -------------------------- | --------- | --------------------------- | ---- | --------------- | --------- |
| 0                          | UV_Value  | Under-voltage Value         | RW   | (10.0,3.4E38]   | float     |
| 1                          | KT_Value  | Torque Coefficient          | RW   | [0.0,3.4E38]    | float     |
| 2                          | OT_Value  | Over-temperature            | RW   | [80.0,200)      | float     |
| 3                          | OC_Value  | Over-current Value          | RW   | (0.0,1.0)       | float     |
| 4                          | ACC       | Acceleration                | RW   | (0.0,3.4E38)    | float     |
| 5                          | DEC       | Deceleration                | RW   | [-3.4E38,0.0)   | float     |
| 6                          | MAX_SPD   | Maximum Speed               | RW   | (0.0,3.4E38]    | float     |
| 7                          | MST_ID    | Feedback ID                 | RW   | [0,0x7FF]       | uint32    |
| 8                          | ESC_ID    | Receive ID                  | RW   | [0,0x7FF]       | uint32    |
| 9                          | TIMEOUT   | Timeout Alarm Time          | RW   | [0,2^32-1]      | uint32    |
| 10                         | CTRL_MODE | Control Mode                | RW   | [1,4]           | uint32    |
| 11                         | Damp      | Motor Damping Coeff.        | RO   | /               | float     |
| 12                         | Inertia   | Motor Inertia               | RO   | /               | float     |
| 13                         | hw_ver    | Reserved                    | RO   | /               | uint32    |
| 14                         | sw_ver    | Software Version            | RO   | /               | uint32    |
| 15                         | SN        | Reserved                    | RO   | /               | uint32    |
| 16                         | NPP       | Motor Pole Pairs            | RO   | /               | uint32    |
| 17                         | Rs        | Motor Phase Resistance      | RO   | /               | float     |
| 18                         | Ls        | Motor Phase Inductance      | RO   | /               | float     |
| 19                         | Flux      | Motor Flux Value            | RO   | /               | float     |
| 20                         | Gr        | Gear Reduction Ratio        | RO   | /               | float     |
| 21                         | PMAX      | Position Mapping Max        | RW   | (0.0,3.4E38]    | float     |
| 22                         | VMAX      | Speed Mapping Max           | RW   | (0.0,3.4E38]    | float     |
| 23                         | TMAX      | Torque Mapping Max          | RW   | (0.0,3.4E38]    | float     |
| 24                         | I_BW      | Current Loop Bandwidth      | RW   | [100.0,10000.0] | float     |
| 25                         | KP_ASR    | Speed Loop Kp               | RW   | [0.0,3.4E38]    | float     |
| 26                         | KI_ASR    | Speed Loop Ki               | RW   | [0.0,3.4E38]    | float     |
| 27                         | KP_APR    | Position Loop Kp            | RW   | [0.0,3.4E38]    | float     |
| 28                         | KI_APR    | Position Loop Ki            | RW   | [0.0,3.4E38]    | float     |
| 29                         | OV_Value  | Over-voltage Value          | RW   | TBD             | float     |
| 30                         | GREF      | Gear Torque Efficiency      | RW   | (0.0,1.0]       | float     |
| 31                         | Deta      | Speed Loop Damping Coeff.   | RW   | [1.0,30.0]      | float     |
| 32                         | V_BW      | Speed Loop Filter Bandwidth | RW   | (0.0,500.0)     | float     |
| 33                         | IQ_c1     | Current Loop Gain           | RW   | [100.0,10000.0] | float     |
| 34                         | VL_c1     | Speed Loop Gain             | RW   | (0.0,10000.0]   | float     |
| 35                         | can_br    | CAN Baud Rate Code          | RW   | [0,4]           | uint32    |
| 36                         | sub_ver   | Sub-version                 | RO   |                 | uint32    |
| 50                         | u_off     | U Phase Offset              | RO   |                 | float     |
| 51                         | v_off     | V Phase Offset              | RO   |                 | float     |
| 52                         | k1        | Compensation Factor 1       | RO   |                 | float     |
| 53                         | k2        | Compensation Factor 2       | RO   |                 | float     |
| 54                         | m_off     | Angle Offset                | RO   |                 | float     |
| 55                         | dir       | Direction                   | RO   |                 | float     |
| 80                         | p_m       | Motor Position              | RO   |                 | float     |
| 81                         | xout      | Output Shaft Position       | RO   |                 | float     |
