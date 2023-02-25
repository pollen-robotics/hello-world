"""
Asleep behavior definition.

Asleep behavior is used to wait between behaviors.
"""

import asyncio
import time
from .breathing import ArmBreathing

import numpy as np

from reachy_sdk.trajectory import goto_async

from . import Behavior
from .player import playsound


class Asleep(Behavior):
    """
    Asleep class.

    Uses: right_arm, left_arm, head, sound
    Dependencies to other behaviors: ArmBreathing
    """

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy=reachy, sub_behavior=sub_behavior)
        self.left_pos = [0, 0, 0, 0, 0, 0, 0]
        self.right_pos = [0, 0, 0, 0, 0, 0, 0]
        self.joint_names = list(reachy.l_arm.joints.values()) + list(self.reachy.r_arm.joints.values())

        self.inhale = 'sounds/inhaling.mp3'

        self.playsIsOk0 = True
        self.playsIsOk = False
        self.playsIsOk2 = False

    async def run(self):
        self.reachy.turn_on('reachy')

        breathing = ArmBreathing(name='arm_breathing', reachy=self.reachy, fundamental_frequency=0.3, phase=-np.pi/4)

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
        self.reachy.turn_off_smoothly('head')

        self.reachy.head.l_antenna.compliant = False
        self.reachy.head.r_antenna.compliant = False

        tic = time.time()

        await breathing.start()

        while time.time() - tic < 10.0:
            t = time.time() - tic

            pos = 20 * np.sin(2 * np.pi * 0.3 * t)
            self.reachy.head.l_antenna.goal_position = 70 + pos
            self.reachy.head.r_antenna.goal_position = -70 - pos

            if (time.time() - tic > 1.2) and self.playsIsOk0:
                playsound(self.inhale, block=False)
                self.playsIsOk0 = False
                self.playsIsOk = True

            if (time.time() - tic > 4.4) and self.playsIsOk:
                playsound(self.inhale, block=False)
                self.playsIsOk = False
                self.playsIsOk2 = True

            if (time.time() - tic > 7.8) and self.playsIsOk2:
                playsound(self.inhale, block=False)
                self.playsIsOk2 = False

            await asyncio.sleep(0.01)

        await breathing.stop()

        self.playsIsOk0 = True
        self.playsIsOk = False
        self.playsIsOk2 = False
