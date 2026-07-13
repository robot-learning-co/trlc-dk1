"""
End-effector (Cartesian) teleoperation for the TRLC-DK1.

Same hand-rolled loop as ``teleop.py``, but the leader's joint action is routed
through LeRobot's EE processor pipelines:

    leader joints --(FK)--> ee.*   (this is what a dataset would record)
                  --(send)--> follower joints

With ``teleop_source="joint_leader"`` the leader already provides joint targets,
so no inverse kinematics runs here — the follower is commanded with the leader's
joints directly and ``ee.*`` is only computed for logging/recording. IK enters
only for EE-space policy inference or delta devices.

Run:  python examples/teleop_ee.py
"""

from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig
from lerobot_robot_trlc_dk1.ee_processors import make_dk1_processors
import time


follower_config = DK1FollowerConfig(
    port="/dev/tty.usbmodem00000000050C1",
    control_mode="impedance",
)

leader_config = DK1LeaderConfig(
    port="/dev/tty.usbmodemE072A1F88AAC1"
)

leader = DK1Leader(leader_config)
leader.connect()

follower = DK1Follower(follower_config)
follower.connect()

# One call builds the three pipelines (and the MuJoCo DK1Kinematics inside).
teleop_action_processor, robot_action_processor, robot_observation_processor = make_dk1_processors(
    action_space="ee",
    teleop_source="joint_leader",
    keep_joints=True,
    drive_via_ik=True,     # close the loop through Cartesian space: leader joints -> FK -> ee.* -> IK -> follower
    ik_seed_previous=True,  # warm-start from the previous command (continuity; enables the rate limit)
    ik_max_step_rad=0.1,   # in-solver per-step trust region ~2.4 rad/s at 60 Hz; prevents jumps
)

freq = 60  # Hz

try:
    while True:
        obs = follower.get_observation()                       # joint_*.pos (rad) + gripper.pos
        obs = robot_observation_processor(obs)                 # + ee.* (FK)

        leader_action = leader.get_action()                    # joint_*.pos (rad) + gripper.pos
        ee_action = teleop_action_processor((leader_action, obs))   # + ee.* (what a dataset records)
        robot_action = robot_action_processor((ee_action, obs))     # joints for send_action

        follower.send_action(robot_action)
        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping EE teleop...")
    leader.disconnect()
    follower.disconnect()
