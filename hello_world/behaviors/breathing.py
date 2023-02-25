import asyncio
import time

import numpy as np

from reachy_sdk.trajectory import goto_async


from . import Behavior


class ArmBreathing(Behavior):
    """
    ArmBreathing class.

    Makes Reachy discreetly swings its arms at a given frequency.

    Uses: right_arm, left_arm
    Dependencies to other behaviors: none
    """
    def __init__(
            self,
            name: str,
            reachy,
            sub_behavior: bool = False,
            fundamental_frequency: float = 0.3,
            phase: float = 0
            ) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)
        self.fundamental_frequency = fundamental_frequency
        self.phase = phase

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        await goto_async({
            self.reachy.l_arm.l_shoulder_pitch: 0.0,
            self.reachy.l_arm.l_shoulder_roll: 0.0,
            self.reachy.l_arm.l_arm_yaw: 0.0,
            self.reachy.l_arm.l_elbow_pitch: 0.0,
            self.reachy.l_arm.l_forearm_yaw: 0.0,
            self.reachy.l_arm.l_wrist_pitch: 0.0,
            self.reachy.l_arm.l_wrist_roll: 0.0,
            self.reachy.l_arm.l_gripper: 0.0,
            self.reachy.r_arm.r_shoulder_pitch: 0.0,
            self.reachy.r_arm.r_shoulder_roll: 0.0,
            self.reachy.r_arm.r_arm_yaw: 0.0,
            self.reachy.r_arm.r_elbow_pitch: 0.0,
            self.reachy.r_arm.r_forearm_yaw: 0.0,
            self.reachy.r_arm.r_wrist_pitch: 0.0,
            self.reachy.r_arm.r_wrist_roll: 0.0,
            self.reachy.r_arm.r_gripper: 0.0,
            },
            duration=1.0,
        )

        t0 = time.time()

        while True:
            pos = 4 * np.sin(2 * np.pi * self.fundamental_frequency * (time.time()-t0) + self.phase)
            self.reachy.r_arm.r_arm_yaw.goal_position = pos
            self.reachy.l_arm.l_arm_yaw.goal_position = -pos
            pos2 = 1.5 * np.sin(2 * np.pi * self.fundamental_frequency * (time.time()-t0) + np.pi + self.phase)
            self.reachy.r_arm.r_shoulder_roll.goal_position = pos2
            self.reachy.l_arm.l_shoulder_roll.goal_position = -pos2
            pos3 = 3 * np.sin(2 * np.pi * self.fundamental_frequency/2 * (time.time()-t0) + np.pi + self.phase)
            self.reachy.r_arm.r_forearm_yaw.goal_position = pos3
            self.reachy.l_arm.l_forearm_yaw.goal_position = -pos3
            pos4 = 4 * np.sin(2 * np.pi * self.fundamental_frequency * (time.time()-t0) + np.pi + self.phase)
            self.reachy.r_arm.r_gripper.goal_position = -pos4
            self.reachy.l_arm.l_gripper.goal_position = pos4
            await asyncio.sleep(0.01)

    async def teardown(self):
        return await super().teardown()
