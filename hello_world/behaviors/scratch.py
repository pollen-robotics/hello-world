"""Implement scratch behavior where Reachy use its left arm to scratch its right forearm."""
import asyncio
import numpy as np

from reachy_sdk.trajectory import goto_async, InterpolationMode

from . import Behavior


class Scratch(Behavior):
    """
    Scratch class.

    Makes Reachy scratch its forearm.

    Uses: right_arm, left_arm, head
    Dependencies to other behaviors: none
    """

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Initialize the behavior."""
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.scratch_arm = np.load('movements/scratch.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.r_arm.r_shoulder_pitch,
            self.reachy.r_arm.r_shoulder_roll,
            self.reachy.r_arm.r_arm_yaw,
            self.reachy.r_arm.r_elbow_pitch,
            self.reachy.r_arm.r_forearm_yaw,
            self.reachy.r_arm.r_wrist_pitch,
            self.reachy.r_arm.r_wrist_roll,
            self.reachy.l_arm.l_shoulder_pitch,
            self.reachy.l_arm.l_shoulder_roll,
            self.reachy.l_arm.l_arm_yaw,
            self.reachy.l_arm.l_elbow_pitch,
            self.reachy.l_arm.l_forearm_yaw,
            self.reachy.l_arm.l_wrist_pitch,
            self.reachy.l_arm.l_wrist_roll,
            self.reachy.l_arm.l_gripper,
        ]

    async def run(self):
        """Implement the Scratch behavior."""
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        look_down = self.reachy.head.look_at_async(
            0.5,
            -0.1,
            -0.5,
            duration=1.0,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            })

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -30,
                self.reachy.head.l_antenna: 5,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        first_point = dict(zip(self.recorded_joints, self.scratch_arm[50]))
        # Goes to the start of the trajectory in 1s
        first_pos = goto_async(first_point, duration=1.0)

        await asyncio.gather(
            look_down,
            first_pos,
            traj_antennas,
        )

        for jp_arms in self.scratch_arm[50:]:
            for joint, pos in zip(self.recorded_joints, jp_arms):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency*2))

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -10,
                self.reachy.head.l_antenna: 65,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        hands_back = goto_async({
            self.reachy.l_arm.l_shoulder_pitch: 0.0,
            self.reachy.l_arm.l_shoulder_roll: 0.0,
            self.reachy.l_arm.l_arm_yaw: 0.0,
            self.reachy.l_arm.l_elbow_pitch: 0.0,
            self.reachy.l_arm.l_forearm_yaw: 0.0,
            self.reachy.l_arm.l_wrist_pitch: 0.0,
            self.reachy.l_arm.l_wrist_roll: 0.0},
            duration=1.2,
        )
        watch_arm_head = self.reachy.head.look_at_async(
            0.5,
            -0.1,
            -0.3,
            duration=0.5,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            })
        pose_watch_right_arm = [-21, 7, 38, -102, 27, -10, -4]
        goto_dic = {j: pos for j, pos in zip(self.reachy.r_arm.joints.values(), pose_watch_right_arm)}
        watch_arm_arm = goto_async(goal_positions=goto_dic, duration=0.5)

        await asyncio.gather(
            watch_arm_head,
            watch_arm_arm,
            hands_back,
            traj_antennas,
        )

        await asyncio.sleep(0.3)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: 0,
                self.reachy.head.l_antenna: 0,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

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
        hands_back = goto_async({
            self.reachy.r_arm.r_shoulder_pitch: 0.0,
            self.reachy.r_arm.r_shoulder_roll: 0.0,
            self.reachy.r_arm.r_arm_yaw: 0.0,
            self.reachy.r_arm.r_elbow_pitch: 0.0,
            self.reachy.r_arm.r_forearm_yaw: 0.0,
            self.reachy.r_arm.r_wrist_pitch: 0.0,
            self.reachy.r_arm.r_wrist_roll: 0.0},
            duration=1.2,
        )

        await asyncio.gather(
            traj_antennas,
            look_back,
            hands_back,
        )

        self.reachy.turn_off_smoothly('r_arm')
        self.reachy.turn_off_smoothly('l_arm')

    async def teardown(self):
        """Use the teardown method from parent class."""
        return await super().teardown()
