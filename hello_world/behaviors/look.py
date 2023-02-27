"""
Look behaviors.

Behaviors making Reachy look at something.
"""

import asyncio
import numpy as np

from reachy_sdk.trajectory import goto_async, InterpolationMode

from . import Behavior


class LookHand(Behavior):
    """
    LookHand class.

    Makes Reachy move its gripper and watch at it.

    Uses: right_arm, head
    Dependencies to other behaviors: none
    """

    async def run(self):
        """Implement the LookHand behavior."""
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0

        base_pos_right = [-1.73, -3.67, -0.57, -68.44, 4.0, -29.67, -4.84]

        A = self.reachy.r_arm.forward_kinematics(joints_position=base_pos_right)

        x = np.random.randint(25, 35) / 100
        y = np.random.randint(-40, -10) / 100
        z = np.random.randint(-10, 0) / 100

        pos = [x, y, z]

        B = A.copy()
        for i in range(3):
            B[i][3] = pos[i]

        JB = self.reachy.r_arm.inverse_kinematics(B, q0=base_pos_right)

        traj_right = goto_async(
            goal_positions={j: p for j, p in zip(self.reachy.r_arm.joints.values(), JB[:-2])},
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        traj_head = self.reachy.head.look_at_async(
            x,
            y,
            z-0.1,
            duration=1,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            })

        await asyncio.gather(
            traj_right,
            traj_head,
        )

        s = 1
        nb_iter = np.random.randint(2, 5)
        for i in range(nb_iter):
            s = -s
            await goto_async(
                goal_positions={self.reachy.r_arm.r_forearm_yaw: self.reachy.r_arm.r_forearm_yaw.present_position + 30*s},
                duration=0.5,
            )
            await asyncio.sleep(0.1)

        nb_iter = np.random.randint(2, 3)
        for i in range(nb_iter):
            self.reachy.r_arm.r_gripper.goal_position = 50
            await asyncio.sleep(0.1)
            self.reachy.r_arm.r_gripper.goal_position = 0
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.3)

        look_back = self.reachy.head.look_at_async(
            0.5,
            0.0,
            0.0,
            duration=1.5,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            })
        hand_back = goto_async({
            self.reachy.r_arm.r_shoulder_pitch: 0.0,
            self.reachy.r_arm.r_shoulder_roll: 0.0,
            self.reachy.r_arm.r_arm_yaw: 0.0,
            self.reachy.r_arm.r_elbow_pitch: 0.0,
            self.reachy.r_arm.r_forearm_yaw: 0.0,
            self.reachy.r_arm.r_wrist_pitch: 0.0,
            self.reachy.r_arm.r_wrist_roll: 0.0
        },
            duration=2.0,
        )

        await asyncio.gather(
            look_back,
            hand_back,
        )

        self.reachy.turn_off_smoothly('r_arm')
