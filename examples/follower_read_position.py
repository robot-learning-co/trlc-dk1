from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *

import time

follower_config = DK1FollowerConfig(
    port="/dev/ttyACM1",
)
follower = DK1Follower(follower_config)

follower.connect()

for key, motor in follower.motors.items():
    follower.control.disable(motor)
    
try:
    while True:
        print(follower.get_observation())
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping read position...")
    follower.disconnect()
