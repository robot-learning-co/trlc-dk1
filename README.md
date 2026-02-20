<p align="center">
    <img src="media/xray.jpg">
</p>
<p align="center">
    <a href="https://docs.robot-learning.co/">
        <img src="https://img.shields.io/badge/Documentation-📕-blue" alt="Chat on Discord"></a>
    <a href="https://discord.gg/PTZ3CN5WkJ">
        <img src="https://img.shields.io/discord/1409155673572249672?color=7289DA&label=Discord&logo=discord&logoColor=white"></a>
    <a href="https://x.com/JannikGrothusen">
        <img src="https://img.shields.io/twitter/follow/Jannik?style=social"></a>
    <a href="https://www.robot-learning.co/">
        <img src=https://img.shields.io/badge/Order%20a%20kit-8A2BE2></a>
</p>

<h1 align="center">An Open Source Dev Kit for AI-native Robotics</h1>
<p align="center">by The Robot Learning Company</p>

## Demo

<p align="center">
    <img src="media/demo.gif">
</p>

## CAD

<table align="center">
<tr>
<td width="50%">
<a href="https://github.com/robot-learning-co/trlc-dk1/blob/main/hardware/TRLC-DK1-Follower_v0.3.0.step" target="_blank">
TRLC-DK1 v0.3.0 Follower CAD<br>
<img src="media/follower_cad.png" width="100%">
</a>
</td>
<td width="50%">
<a href="https://a360.co/481PSQH" target="_blank">
TRLC-DK1 v0.2.0 Leader CAD<br>
<img src="media/leader_cad.png" width="100%">
</a>
</td>
</tr>
</table>
Copyright 2025-2026 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.

## Installation

```
git clone https://github.com/robot-learning-co/trlc-dk1.git
uv venv
uv pip install -e .
uv pip install pyrealsense2
```
This repo uses [LeRobot's plugin conventions](https://huggingface.co/docs/lerobot/integrate_hardware#using-your-own-lerobot-devices-) to be automatically detected by a LeRobot installation in the same Python environment.

### Permissions

To access USB cameras and serial ports (robot arms), your user must be in the `video` and `dialout` groups:

```
sudo usermod -aG video,dialout $(whoami)
```

Log out and log back in for the change to take effect.


## Examples

Use [LeRobot's CLI](https://huggingface.co/docs/lerobot/il_robots) to identify your teleop, robot, and camera ports:

```
uv run lerobot-find-port
uv run lerobot-find-cameras
```
The port values in the examples below correspond to the autonomy-eagle setup in Aera41 (don't change the plug setup there). 

<details>
<summary>Example I: Single Arm Teleoperation
</summary>

```bash
uv run lerobot-teleoperate \
    --robot.type=dk1_follower \
    --robot.port=/dev/ttyACM3 \
    --robot.joint_velocity_scaling=0.2 \
    --robot.disable_torque_on_disconnect=true \
    --teleop.type=dk1_leader \
    --teleop.port=/dev/ttyACM1 \
    --robot.cameras="{ 
        right_wrist: {type: opencv, index_or_path: /dev/video0, width: 1280, height: 720, fps: 60, rotation: 180, fourcc: "MJPG"}
      }" \
    --display_data=true
```
</details>

<details>
<summary>Example II: Bimanual Teleoperation
</summary>

```bash
uv run lerobot-teleoperate \
    --robot.type=bi_dk1_follower \
    --robot.right_arm_port=/dev/ttyACM3 \
    --robot.left_arm_port=/dev/ttyACM2 \
    --robot.joint_velocity_scaling=0.2 \
    --robot.disable_torque_on_disconnect=true \
    --teleop.type=bi_dk1_leader \
    --teleop.right_arm_port=/dev/ttyACM1 \
    --teleop.left_arm_port=/dev/ttyACM0 \
    --robot.cameras="{
        right_wrist: {type: opencv, index_or_path: /dev/video0, width: 1280, height: 720, fps: 60, rotation: 180, fourcc: "MJPG"},
        left_wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 60, rotation: 180, fourcc: "MJPG"}
      }" \
    --display_data=true
```
</details>

<details>
<summary>Example III: Bimanual Recording
</summary>

```bash
uv run lerobot-record \
    --robot.type=bi_dk1_follower \
    --robot.right_arm_port=/dev/ttyACM3 \
    --robot.left_arm_port=/dev/ttyACM2 \
    --robot.joint_velocity_scaling=1.0 \
    --robot.disable_torque_on_disconnect=true \
    --teleop.type=bi_dk1_leader \
    --teleop.right_arm_port=/dev/ttyACM1 \
    --teleop.left_arm_port=/dev/ttyACM0 \
    --robot.cameras="{ 
        head: {type: opencv, index_or_path: /dev/video0, width: 960, height: 540, fps: 60, fourcc: "MJPG"},
        right_wrist: {type: opencv, index_or_path: /dev/video2, width: 960, height: 540, fps: 60, rotation: 180, fourcc: "MJPG"},
        right_wrist: {type: opencv, index_or_path: /dev/video0, width: 960, height: 540, fps: 60, rotation: 180, fourcc: "MJPG"},
        left_wrist: {type: opencv, index_or_path: /dev/video2, width: 960, height: 540, fps: 60, rotation: 180, fourcc: "MJPG"},
      }" \
    --dataset.repo_id=$USER/my_test_dataset \
    --dataset.push_to_hub=false \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=20 \
    --dataset.single_task="Test the LeRobot recording pipeline."
```
</details>

## URDF (v0.2)

<p align="center">
    <img src="https://github.com/andreaskoepf/trlc-dk1-follower-urdf/blob/main/assets/dk1_vsual_right.png">
</p>

A URDF file was developed by community member Andreas Köpf:
[andreaskoepf/trlc-dk1-follower-urdf](https://github.com/andreaskoepf/trlc-dk1-follower-urdf)

The impedance controller requires a local clone of the URDF on the `fix/urdf` branch:

```
git clone -b fix/urdf git@github.com:rtkg/trlc-dk1-follower-urdf.git
```

## Acknowledgements

- [GELLO](https://wuphilipp.github.io/gello_site/) by Philipp Wu et al.
- [Low-Cost Robot Arm](https://github.com/AlexanderKoch-Koch/low_cost_robot) by Alexander Koch
- [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace, Inc.
- [SO-100](https://github.com/TheRobotStudio/SO-ARM100) by TheRobotStudio
- [OpenArm](https://openarm.dev/) by Enactic, Inc.
