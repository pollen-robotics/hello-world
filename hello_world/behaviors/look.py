import asyncio
import numpy as np

from reachy_sdk.trajectory import goto_async, InterpolationMode

from . import Behavior


class LookAt(Behavior):
    async def run(self):
        base_pos_right = [-1.73, -3.67, -0.57, -68.44, 4.0, -29.67, -4.84, -47.14]
        goto_dic = {j: pos for j, pos in zip(self.reachy.r_arm.joints.values(), base_pos_right)}
        arm_down = goto_async(goal_positions=goto_dic, duration=1.0)
        first_look_at = self.reachy.head.look_at_async(x=0.5, y=-0.5, z=0.1, duration=1.1)
        await asyncio.gather(arm_down, first_look_at)

        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        # await self.reachy.head.look_at_async(x=0.5, y=-0.5, z=0.1, duration=1.1)
        await asyncio.sleep(0.2)
        await self.reachy.head.look_at_async(x=0.5, y=0, z=-0.4, duration=1.1)
        await asyncio.sleep(0.2)
        await self.reachy.head.look_at_async(x=0.5, y=0.3, z=-0.3, duration=1.1)
        await asyncio.sleep(0.2)
        await self.reachy.head.look_at_async(x=0.5, y=0, z=0, duration=1.1)


class LookAt2(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.look_at = np.load('movements/look_at2.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        first_point = dict(zip(self.recorded_joints, self.look_at[0]))
        # Goes to the start of the trajectory in 3s
        await goto_async(first_point, duration=1.0)

        for jp_head in self.look_at:
            for joint, pos in zip(self.recorded_joints, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

    async def teardown(self):
        return await super().teardown()


class LookAt3(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.look_at = np.load('movements/look_at3.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        first_point = dict(zip(self.recorded_joints, self.look_at[0]))
        # Goes to the start of the trajectory in 3s
        await goto_async(first_point, duration=1.0)

        for jp_head in self.look_at:
            for joint, pos in zip(self.recorded_joints, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

    async def teardown(self):
        return await super().teardown()


class LookAt4(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.look_at = np.load('movements/look_at4.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        first_point = dict(zip(self.recorded_joints, self.look_at[0]))
        # Goes to the start of the trajectory in 3s
        await goto_async(first_point, duration=1.0)

        for jp_head in self.look_at:
            for joint, pos in zip(self.recorded_joints, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

    async def teardown(self):
        return await super().teardown()


class LookAt5(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.look_at = np.load('movements/look_at5.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        first_point = dict(zip(self.recorded_joints, self.look_at[0]))
        # Goes to the start of the trajectory in 3s
        await goto_async(first_point, duration=1.0)

        for jp_head in self.look_at:
            for joint, pos in zip(self.recorded_joints, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

    async def teardown(self):
        return await super().teardown()


class LookAt6(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.look_at = np.load('movements/look_at6.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        first_point = dict(zip(self.recorded_joints, self.look_at[0]))
        # Goes to the start of the trajectory in 3s
        await goto_async(first_point, duration=1.0)

        for jp_head in self.look_at:
            for joint, pos in zip(self.recorded_joints, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

    async def teardown(self):
        return await super().teardown()


class LookAt7(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.look_at = np.load('movements/look_at7.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        first_point = dict(zip(self.recorded_joints, self.look_at[0]))
        # Goes to the start of the trajectory in 3s
        await goto_async(first_point, duration=1.0)

        for jp_head in self.look_at:
            for joint, pos in zip(self.recorded_joints, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

    async def teardown(self):
        return await super().teardown()


class LookAt8(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.look_at = np.load('movements/look_at8.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        first_point = dict(zip(self.recorded_joints, self.look_at[0]))
        # Goes to the start of the trajectory in 3s
        await goto_async(first_point, duration=1.0)

        for jp_head in self.look_at:
            for joint, pos in zip(self.recorded_joints, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

    async def teardown(self):
        return await super().teardown()


class LookHand(Behavior):
    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0

        base_pos_right = [-1.73, -3.67, -0.57, -68.44, 4.0, -29.67, -4.84]
        # goto_dic = {j: pos for j, pos in zip(self.reachy.r_arm.joints.values(), base_pos_right)}
        # await goto_async(goal_positions=goto_dic, duration=1.0)
        # await self.reachy.head.look_at_async(
        #     x=0.5,
        #     y=0.3,
        #     z=-0.3,
        #     duration=1.1,
        #     starting_positions={
        #         self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
        #         self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
        #         self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
        #     })

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
