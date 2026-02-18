from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig
from lerobot_robot_trlc_dk1.bi_leader import BiDK1Leader, BiDK1LeaderConfig
import time
import logging

from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

init_logging()

follower_config = BiDK1FollowerConfig(
    left_arm_port="/dev/ttyACM1",
    right_arm_port="/dev/ttyACM0",
    joint_velocity_scaling=0.2,
)

leader_config = BiDK1LeaderConfig( 
    left_arm_port="/dev/ttyACM3",
    right_arm_port="/dev/ttyACM2",
)

leader = BiDK1Leader(leader_config)
leader.connect()

follower = BiDK1Follower(follower_config)
follower.connect()

freq = 200 # Hz

try:
    while True:
        action = leader.get_action()
        follower.send_action(action)    
        time.sleep(1/freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
    leader.disconnect()
    follower.disconnect()
