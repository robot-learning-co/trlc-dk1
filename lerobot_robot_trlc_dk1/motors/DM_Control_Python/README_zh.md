# 达妙 Python 库

语言: [中文](README_zh.md) | [ENGLISH](README.md) 

该库支持 **macOS**、**Linux** 和 **Windows** 三个平台。

🎉 **欢迎加入 QQ 群：677900232** 进行达妙电机技术交流。  
🛒 **点击访问达妙智能控制企业店**：[达妙店铺](https://shop290016675.taobao.com/?spm=pc_detail.29232929/evo365560b447259.shop_block.dshopinfo.59f47dd6w4Z4dX)

* * *

## 目录

1. [引用达妙库](#1-%E5%BC%95%E7%94%A8%E8%BE%BE%E5%A6%99%E5%BA%93)
2. [定义电机控制对象](#2-%E5%AE%9A%E4%B9%89%E7%94%B5%E6%9C%BA%E6%8E%A7%E5%88%B6%E5%AF%B9%E8%B1%A1)
3. [电机状态](#3-%E7%94%B5%E6%9C%BA%E7%8A%B6%E6%80%81)
    * [添加电机](#31-%E6%B7%BB%E5%8A%A0%E7%94%B5%E6%9C%BA)
    * [使能电机](#32-%E4%BD%BF%E8%83%BD%E7%94%B5%E6%9C%BA)
    * [设置零点](#33-%E8%AE%BE%E7%BD%AE%E9%9B%B6%E7%82%B9)
    * [失能电机](#34-%E5%A4%B1%E8%83%BD%E7%94%B5%E6%9C%BA)
    * [获取电机状态](#35-%E8%8E%B7%E5%8F%96%E7%94%B5%E6%9C%BA%E7%8A%B6%E6%80%81)
4. [电机控制模式](#4-%E7%94%B5%E6%9C%BA%E6%8E%A7%E5%88%B6%E6%A8%A1%E5%BC%8F)
    * [MIT 模式](#41-mit-%E6%A8%A1%E5%BC%8F)
    * [位置速度模式](#42-%E4%BD%8D%E7%BD%AE%E9%80%9F%E5%BA%A6%E6%A8%A1%E5%BC%8F)
    * [速度模式](#43-%E9%80%9F%E5%BA%A6%E6%A8%A1%E5%BC%8F)
    * [力位混合模式](#44-%E5%8A%9B%E4%BD%8D%E6%B7%B7%E5%90%88%E6%A8%A1%E5%BC%8F)
5. [电机状态读取](#5-%E7%94%B5%E6%9C%BA%E7%8A%B6%E6%80%81%E8%AF%BB%E5%8F%96)
6. [电机内部参数更改](#6-%E7%94%B5%E6%9C%BA%E5%86%85%E9%83%A8%E5%8F%82%E6%95%B0%E6%9B%B4%E6%94%B9)
    * [控制模式更改](#61-%E6%8E%A7%E5%88%B6%E6%A8%A1%E5%BC%8F%E6%9B%B4%E6%94%B9)
    * [保存参数](#62-%E4%BF%9D%E5%AD%98%E5%8F%82%E6%95%B0)
    * [读取内部寄存器](#63-%E8%AF%BB%E5%8F%96%E5%86%85%E9%83%A8%E5%AF%84%E5%AD%98%E5%99%A8)
    * [修改内部寄存器](#64-%E4%BF%AE%E6%94%B9%E5%86%85%E9%83%A8%E5%AF%84%E5%AD%98%E5%99%A8)

* * *

## 1. 引用达妙库

确保 `DM_CAN.py` 文件在项目文件夹中，使用以下方式导入：

```python
from DM_CAN import *
```

### 安装依赖

该库依赖以下 Python 库：`serial` 和 `numpy`，请使用以下命令安装：

```bash
pip install -r requirements.txt
```

* * *

## 2. 定义电机控制对象

使用以下代码定义电机对象：

```python
Motor1 = Motor(DM_Motor_Type.DM4310, 0x01, 0x11)
Motor2 = Motor(DM_Motor_Type.DM4310, 0x02, 0x12)
Motor3 = Motor(DM_Motor_Type.DM4310, 0x03, 0x13)
```

* **第一个参数**：电机类型
* **第二个参数**：SlaveID（电机的CAN ID）
* **第三个参数**：MasterID（主机ID，建议不为 `0x00`，且不与 SlaveID 冲突）

### 配置串口通信

```python
serial_device = serial.Serial('COM8', 921600, timeout=0.5)
MotorControl1 = MotorControl(serial_device)
```

* * *

## 3. 电机状态

### 3.1 添加电机

将电机对象添加到控制对象中：

```python
MotorControl1.addMotor(Motor1)
MotorControl1.addMotor(Motor2)
MotorControl1.addMotor(Motor3)
```

### 3.2 使能电机

使能电机前，建议先设置好所有电机参数：

```python
MotorControl1.enable(Motor1)
MotorControl1.enable(Motor2)
MotorControl1.enable(Motor3)
```

如果是旧版固件，需指定控制模式：

```python
MotorControl1.enable_old(Motor1, Control_Type.MIT)
MotorControl1.enable_old(Motor2, Control_Type.POS_VEL)
MotorControl1.enable_old(Motor3, Control_Type.VEL)
```

### 3.3 设置零点

将电机移动到所需零点位置（在失能状态下），然后执行：

```python
MotorControl1.set_zero_position(Motor3)
MotorControl1.set_zero_position(Motor6)
```

### 3.4 失能电机

```python
MotorControl1.disable(Motor3)
MotorControl1.disable(Motor6)
```

### 3.5 获取电机状态

通过以下代码可以获取电机的实时状态（位置、速度、力矩等）：

```python
MotorControl1.refresh_motor_status(Motor1)
print("Motor1:", "POS:", Motor1.getPosition(), "VEL:", Motor1.getVelocity(), "TORQUE:", Motor1.getTorque())
```

* * *

## 4. 电机控制模式

### 4.1 MIT 模式

使用 MIT 模式控制电机：

```python
MotorControl1.controlMIT(Motor1, 50, 0.3, 0, 0, 0)
```

### 4.2 位置速度模式

通过以下代码控制电机的目标位置和速度：

```python
q = math.sin(time.time())
MotorControl1.control_Pos_Vel(Motor1, q*10, 2)
```

### 4.3 速度模式

控制电机的速度：

```python
q = math.sin(time.time())
MotorControl1.control_Vel(Motor1, q*5)
```

### 4.4 力位混合模式

控制电机的目标位置和力：

```python
MotorControl1.control_pos_force(Motor1, 10, 1000, 100)
```

* * *

## 5. 电机状态读取

电机状态保存在对应的电机对象中，可以通过以下函数读取：

```python
vel = Motor1.getVelocity()    # 获得电机速度
pos = Motor1.getPosition()    # 获得电机位置
tau = Motor1.getTorque()      # 获得电机输出力矩
```

* * *

## 6. 电机内部参数更改

### 6.1 控制模式更改

可以在线修改电机的控制模式：

```python
if MotorControl1.switchControlMode(Motor1, Control_Type.POS_VEL):
    print("切换到 POS_VEL 模式成功")
if MotorControl1.switchControlMode(Motor2, Control_Type.VEL):
    print("切换到 VEL 模式成功")
```

### 6.2 保存参数

修改参数后，需要保存到电机的 flash 中：

```python
MotorControl1.save_motor_param(Motor1)
```

### 6.3 读取内部寄存器

读取电机内部寄存器的参数：

```python
print("PMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.PMAX))
print("MST_ID:", MotorControl1.read_motor_param(Motor1, DM_variable.MST_ID))
print("VMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.VMAX))
print("TMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.TMAX))
```

### 6.4 修改内部寄存器

修改电机内部寄存器的参数（此更改在掉电后会恢复）：

```python
if MotorControl1.change_motor_param(Motor1, DM_variable.KP_APR, 54):
   print("写入成功")
```

* * *