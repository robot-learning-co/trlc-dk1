from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig
import time

leader_config = DK1LeaderConfig(
    port="/dev/tty.usbmodem5A460819651"
)

leader = DK1Leader(leader_config)
leader.connect()


while True:
    action = leader.get_action()
    print(action)
    time.sleep(0.02)