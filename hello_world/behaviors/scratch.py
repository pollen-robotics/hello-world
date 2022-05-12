import asyncio
import numpy as np

from reachy_sdk.trajectory import goto_async, InterpolationMode

from . import Behavior


class Scratch(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
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
        # Goes to the start of the trajectory in 3s
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
        # traj_head = self.reachy.head.look_at_async(
        #     0.5,
        #     -0.1,
        #     0.0,
        #     duration=1.0,
        #     starting_positions={
        #         self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
        #         self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
        #         self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
        #     })
        # base_pos_right = [-1.73, -3.67, -0.57, -68.44, 4.0, -29.67, -4.84, -47.14]
        # goto_dic = {j: pos for j, pos in zip(self.reachy.r_arm.joints.values(), base_pos_right)}
        # arm_down = goto_async(goal_positions=goto_dic, duration=1.0)
        # await asyncio.gather(
        #     traj_head,
        #     traj_antennas,
        #     arm_down,
        # )

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
        return await super().teardown()


class Scratch2(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.scratch_shoulder = np.load('movements/scratch_shoulder.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.r_arm.r_shoulder_pitch,
            self.reachy.r_arm.r_shoulder_roll,
            self.reachy.r_arm.r_arm_yaw,
            self.reachy.r_arm.r_elbow_pitch,
            self.reachy.r_arm.r_forearm_yaw,
            self.reachy.r_arm.r_wrist_pitch,
            self.reachy.r_arm.r_wrist_roll,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        look_down = self.reachy.head.look_at_async(0.5, 0.1, -0.5, duration=1.5)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -5,
                self.reachy.head.l_antenna: 15,
            },
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        first_point = dict(zip(self.recorded_joints, self.scratch_shoulder[220]))
        # Goes to the start of the trajectory in 3s
        first_pos = goto_async(first_point, duration=1.5)

        await asyncio.gather(
            look_down,
            first_pos,
            traj_antennas,
        )

        for jp_arms in self.scratch_shoulder[220:540]:
            for joint, pos in zip(self.recorded_joints, jp_arms):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -0,
                self.reachy.head.l_antenna: 0,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        back_head = self.reachy.head.look_at_async(0.5, -0.0, -0.0, duration=1.5)
        base_pos_right = [-1.73, -3.67, -0.57, -68.44, 4.0, -29.67, -4.84, -47.14]
        goto_dic = {j: pos for j, pos in zip(self.reachy.r_arm.joints.values(), base_pos_right)}
        base_pos = goto_async(goal_positions=goto_dic, duration=1.5)

        await asyncio.gather(
            back_head,
            base_pos,
            traj_antennas,
        )

    async def teardown(self):
        return await super().teardown()


class Scratch3(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.scratch_left = np.load('movements/scratch3_l_arm.npy')
        self.scratch_right = np.load('movements/scratch3_r_arm.npy')

        self.sampling_frequency = 100

        self.recorded_joints_left = [
            self.reachy.l_arm.l_shoulder_pitch,
            self.reachy.l_arm.l_shoulder_roll,
            self.reachy.l_arm.l_arm_yaw,
            self.reachy.l_arm.l_elbow_pitch,
            self.reachy.l_arm.l_forearm_yaw,
            self.reachy.l_arm.l_wrist_pitch,
            self.reachy.l_arm.l_wrist_roll,
        ]

        self.recorded_joints_right = [
            self.reachy.r_arm.r_shoulder_pitch,
            self.reachy.r_arm.r_shoulder_roll,
            self.reachy.r_arm.r_arm_yaw,
            self.reachy.r_arm.r_elbow_pitch,
            self.reachy.r_arm.r_forearm_yaw,
            self.reachy.r_arm.r_wrist_pitch,
            self.reachy.r_arm.r_wrist_roll,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        look_down = self.reachy.head.look_at_async(0.5, 0.2, -0.5, duration=1.5)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -20,
                self.reachy.head.l_antenna: 20,
            },
            duration=1.5,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        first_point_l = dict(zip(self.recorded_joints_left, self.scratch_left[100]))
        # Goes to the start of the trajectory in 3s
        first_pos_l = goto_async(first_point_l, duration=1.5)

        first_point_r = dict(zip(self.recorded_joints_right, self.scratch_right[100]))
        # Goes to the start of the trajectory in 3s
        first_pos_r = goto_async(first_point_r, duration=1.5)

        await asyncio.gather(
            look_down,
            first_pos_l,
            first_pos_r,
            traj_antennas,
        )

        for (jp_l_arm, jp_r_arm) in zip(self.scratch_left[100:1000], self.scratch_right[100:1000]):
            for joint, pos in zip(self.recorded_joints_left, jp_l_arm):
                joint.goal_position = pos
            for joint, pos in zip(self.recorded_joints_right, jp_r_arm):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -0,
                self.reachy.head.l_antenna: 0,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        back_head = self.reachy.head.look_at_async(0.5, -0.0, -0.0, duration=1.5)

        last_point_l = dict(zip(self.recorded_joints_left, self.scratch_left[1180]))
        # Goes to the start of the trajectory in 3s
        last_pos_l = goto_async(last_point_l, duration=1.0)

        last_point_r = dict(zip(self.recorded_joints_right, self.scratch_right[1180]))
        # Goes to the start of the trajectory in 3s
        last_pos_r = goto_async(last_point_r, duration=1.0)

        await asyncio.gather(
            back_head,
            traj_antennas,
            last_pos_l,
            last_pos_r
        )

    async def teardown(self):
        return await super().teardown()
