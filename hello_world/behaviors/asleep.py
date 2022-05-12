import asyncio
import time

import numpy as np

from reachy_sdk.trajectory import goto_async


from . import Behavior


class Asleep(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy=reachy, sub_behavior=sub_behavior)
        self.left_pos = [0, 0, 0, 0, 0, 0, 0]
        self.right_pos = [0, 0, 0, 0, 0, 0, 0]
        self.joint_names = list(reachy.l_arm.joints.values()) + list(self.reachy.r_arm.joints.values())

    async def run(self):
        self.reachy.turn_on('reachy')

        goto = goto_async(
            goal_positions={
                joint_name: pos
                for (joint_name, pos) in zip(self.joint_names, self.left_pos + self.right_pos)
            },
            duration=1.5,
        )
        goto_antennas = goto_async({self.reachy.head.l_antenna: 70, self.reachy.head.r_antenna: -70}, duration=1.0)
        look_at = self.reachy.head.look_at_async(x=0.5, y=0, z=-0.3, duration=1.0)
        await asyncio.gather(goto, look_at, goto_antennas)
        self.reachy.turn_off_smoothly('reachy')

        self.reachy.head.l_antenna.compliant = False
        self.reachy.head.r_antenna.compliant = False

        tic = time.time()

        while time.time() - tic < 10.0:
            t = time.time() - tic

            pos = 20 * np.sin(2 * np.pi * 0.3 * t)
            self.reachy.head.l_antenna.goal_position = 70 + pos
            self.reachy.head.r_antenna.goal_position = -70 - pos

            await asyncio.sleep(0.01)
