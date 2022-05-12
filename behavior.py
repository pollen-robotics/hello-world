import asyncio
import numpy as np
from playsound import playsound
import time
from reachy_sdk.trajectory import goto_async
from reachy_sdk.trajectory.interpolation import InterpolationMode

from reachy_face_tracking.detection import Detection


class Behavior:
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        self.name = name
        self._task = None
        self.sub_behavior = sub_behavior
        self.reachy = reachy

    async def start(self):
        # print(f'Start-begin behavior {self.name}')
        await self.setup()
        self._task = asyncio.create_task(self._run(), name=f'behavior_{self.name}')
        # print(f'Start-end behavior {self.name}')
        return self._task

    async def stop(self):
        pass
        # print(f'Stop-begin behavior {self.name}')

        if self._task is not None:
            self._task.cancel()
            await self._task

        # print(f'Stop-end behavior {self.name}')

    async def setup(self):
        pass
        # print(f'Setup behavior {self.name}')

    async def run(self):
        pass

    async def _run(self):
        try:
            await self.run()
        except asyncio.CancelledError:
            # print(f'{self.name} got cancelled')
            if self.sub_behavior:
                raise
        await self.teardown()

    async def teardown(self):
        pass
        # print(f'Teardown behavior {self.name}')

    def is_running(self):
        return self._task is not None and not self._task.done()


class Popote(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.cough_head = np.load('movements/cough_head.npy')
        self.popote = np.load('movements/popote.npy')

        self.sampling_frequency = 100

        self.recorded_joints_head = [
            self.reachy.joints.neck_yaw,
            self.reachy.joints.neck_roll,
            self.reachy.joints.neck_pitch,
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

        first_point_head = dict(zip(self.recorded_joints_head, self.cough_head[0]))
        first_point_arm = dict(zip(self.recorded_joints_right, self.popote[0]))
        # Goes to the start of the trajectory in 3s
        first_pos_head = goto_async(first_point_head, duration=1.0)
        first_pos_arm = goto_async(first_point_arm, duration=1.0)

        await asyncio.gather(
            first_pos_head,
            first_pos_arm,
        )

        for (jp_head, jp_r_arm) in zip(self.cough_head, self.popote):
            for joint, pos in zip(self.recorded_joints_head, jp_head):
                joint.goal_position = pos
            for joint, pos in zip(self.recorded_joints_right, jp_r_arm):
                joint.goal_position = pos

            await asyncio.sleep(1 / self.sampling_frequency)        

    async def teardown(self):
        return await super().teardown()

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
        hand_back = goto_async(
            {self.reachy.r_arm.r_shoulder_pitch: 0.0,
            self.reachy.r_arm.r_shoulder_roll: 0.0,
            self.reachy.r_arm.r_arm_yaw: 0.0,
            self.reachy.r_arm.r_elbow_pitch: 0.0,
            self.reachy.r_arm.r_forearm_yaw: 0.0,
            self.reachy.r_arm.r_wrist_pitch: 0.0,
            self.reachy.r_arm.r_wrist_roll: 0.0},
            duration=2.0,
        )

        await asyncio.gather(
            look_back,
            hand_back,
        )

        self.reachy.turn_off_smoothly('r_arm')

class Lonely(Behavior):
    async def run(self):
        # base_pos_right = [-1.73, -3.67, -0.57, -68.44, 4.0, -29.67, -4.84, -47.14]
        # goto_dic = {j: pos for j, pos in zip(self.reachy.r_arm.joints.values(), base_pos_right)}
        # arm_down = goto_async(goal_positions=goto_dic, duration=1.0)
        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -65,
                self.reachy.head.l_antenna: 10,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        first_look_at = self.reachy.head.look_at_async(
            x=0.5,
            y=-0.3,
            z=0.1,
            duration=1.1,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
                }
            )
        # await asyncio.gather(arm_down, first_look_at, traj_antennas)
        await asyncio.gather(first_look_at, traj_antennas)

        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        await asyncio.sleep(0.2)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -5,
                self.reachy.head.l_antenna: 30,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        look_at_right = self.reachy.head.look_at_async(
            x=0.5,
            y=-0.6,
            z=0.1,
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
            starting_positions ={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
                }
            )
        await asyncio.gather(traj_antennas, look_at_right)
        await asyncio.sleep(0.5)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -10,
                self.reachy.head.l_antenna: 75,
            },
            duration=0.7,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        look_at_left = self.reachy.head.look_at_async(
            0.5,
            0.5,
            -0.1,
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
            starting_positions = {
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
                }
            )
        await asyncio.gather(traj_antennas, look_at_left)
        await asyncio.sleep(0.5)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -150,
                self.reachy.head.l_antenna: 150,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        first_look_down = goto_async(
            self.reachy.head._look_at(x=0.5, y=0.08, z=-0.4),
            duration=1.0,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            },
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        # first_look_down = self.reachy.head.look_at_async(
        #     0.5,
        #     0.08,
        #     -0.4,
        #     duration=1.0,
        #     interpolation_mode=InterpolationMode.MINIMUM_JERK,
        #     starting_positions = {
        #         self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
        #         self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
        #         self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
        #         }
        #     )
        await asyncio.gather(
            traj_antennas,
            first_look_down,
        )
        # await self.reachy.head.look_at_async(
        #     0.5,
        #     -0.08,
        #     -0.4,
        #     duration=1.0,
        #     interpolation_mode=InterpolationMode.MINIMUM_JERK,
        #     starting_positions = {
        #         self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
        #         self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
        #         self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
        #         }
        #     )
        await goto_async(
            self.reachy.head._look_at(x=0.5, y=-0.08, z=-0.4),
            duration=1.0,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            },
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        # await self.reachy.head.look_at_async(
        #     0.5,
        #     0.08,
        #     -0.4,
        #     duration=1.0,
        #     interpolation_mode=InterpolationMode.MINIMUM_JERK,
        #     starting_positions = {
        #         self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
        #         self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
        #         self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
        #         }
        #     )
        await goto_async(
            self.reachy.head._look_at(x=0.5, y=0.08, z=-0.4),
            duration=1.0,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            },
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        await asyncio.sleep(0.2)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: 0,
                self.reachy.head.l_antenna: 0,
            },
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        look_straight = goto_async(
            self.reachy.head._look_at(x=0.5, y=0.2, z=0.0),
            duration=1.0,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            },
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        # look_straight = self.reachy.head.look_at_async(
        #     0.5,
        #     0.2,
        #     0.0,
        #     duration=1.0,
        #     interpolation_mode=InterpolationMode.MINIMUM_JERK,
        #     starting_positions = {
        #         self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
        #         self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
        #         self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
        #         }
        #     )
        await asyncio.gather(
            traj_antennas,
            look_straight,
        )

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
        hands_back = goto_async(
            {self.reachy.l_arm.l_shoulder_pitch: 0.0,
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
        hands_back = goto_async(
            {self.reachy.r_arm.r_shoulder_pitch: 0.0,
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


class Tshirt(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.touch_tshirt = np.load('movements/traj_tshirt.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
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
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        look_down = self.reachy.head.look_at_async(
            0.5,
            0.2,
            -0.5,
            duration=1.0,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            })

        first_point = dict(zip(self.recorded_joints, self.touch_tshirt[100]))
        # Goes to the start of the trajectory in 3s
        first_pos = goto_async(first_point, duration=1.0)

        await asyncio.gather(
            look_down,
            first_pos,
        )

        for jp_arms in self.touch_tshirt[100:]:
            for joint, pos in zip(self.recorded_joints, jp_arms):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency * 1.5))

        # pose_end_arm = [16.11, 7, -9.71, -84.62, 13.67, -5.14, -29.18]
        # goto_dic = {j: pos for j, pos in zip(self.reachy.l_arm.joints.values(), pose_end_arm)}
        # end_move = goto_async(goal_positions=goto_dic, duration=1.0, interpolation_mode=InterpolationMode.MINIMUM_JERK)

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
        hand_back = goto_async(
            {self.reachy.l_arm.l_shoulder_pitch: 0.0,
            self.reachy.l_arm.l_shoulder_roll: 0.0,
            self.reachy.l_arm.l_arm_yaw: 0.0,
            self.reachy.l_arm.l_elbow_pitch: 0.0,
            self.reachy.l_arm.l_forearm_yaw: 0.0,
            self.reachy.l_arm.l_wrist_pitch: 0.0,
            self.reachy.l_arm.l_wrist_roll: 0.0},
            duration=2.0,
        )

        await asyncio.gather(
            look_back,
            hand_back,
        )

        self.reachy.turn_off_smoothly('l_arm')

        # await asyncio.gather(
        #     end_move,
        # )
        await asyncio.gather(
            hand_back,
            look_back,
        )

        await asyncio.sleep(0.3)

    async def teardown(self):
        return await super().teardown()


class SweatHead(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.sweat_head = np.load('movements/sweat_head.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.l_arm.l_shoulder_pitch,
            self.reachy.l_arm.l_shoulder_roll,
            self.reachy.l_arm.l_arm_yaw,
            self.reachy.l_arm.l_elbow_pitch,
            self.reachy.l_arm.l_forearm_yaw,
            self.reachy.l_arm.l_wrist_pitch,
            self.reachy.l_arm.l_wrist_roll,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        # first_point = dict(zip(self.recorded_joints, self.sweat_head[100]))
        # # Goes to the start of the trajectory in 3s
        # await goto_async(first_point, duration=3.0)

        look_down = self.reachy.head.look_at_async(
            0.5,
            -0.2,
            -0.4,
            duration=1.5,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            })

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -180,
                self.reachy.head.l_antenna: 180,
            },
            duration=1.5,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )

        point110 = dict(zip(self.recorded_joints, self.sweat_head[300]))
        # Goes to the start of the trajectory in 3s
        first_pos = goto_async(point110, duration=2.5)

        await asyncio.gather(
            look_down,
            first_pos,
            traj_antennas,
        )

        for jp_arms in self.sweat_head[300:700]:
            for joint, pos in zip(self.recorded_joints, jp_arms):
                joint.goal_position = pos

            await asyncio.sleep(1 / self.sampling_frequency)
        

        look_up = self.reachy.head.look_at_async(
            0.5,
            0,
            0,
            duration=1.0,
            starting_positions={
                self.reachy.head.neck_roll: self.reachy.head.neck_roll.goal_position,
                self.reachy.head.neck_pitch: self.reachy.head.neck_pitch.goal_position,
                self.reachy.head.neck_yaw: self.reachy.head.neck_yaw.goal_position,
            })

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: 0,
                self.reachy.head.l_antenna: 0,
            },
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        point880 = dict(zip(self.recorded_joints, self.sweat_head[980]))
        # Goes to the start of the trajectory in 3s
        # last_pos = goto_async(point880, duration=1.0)
        last_pos = goto_async(
            {self.reachy.l_arm.l_shoulder_pitch: 0.0,
            self.reachy.l_arm.l_shoulder_roll: 0.0,
            self.reachy.l_arm.l_arm_yaw: 0.0,
            self.reachy.l_arm.l_elbow_pitch: 0.0,
            self.reachy.l_arm.l_forearm_yaw: 0.0,
            self.reachy.l_arm.l_wrist_pitch: 0.0,
            self.reachy.l_arm.l_wrist_roll: 0.0},
            duration=2.0,
        )

        await asyncio.gather(
            look_up,
            traj_antennas,
            last_pos,
        )

        for j in self.reachy.l_arm.joints.values():
            j.compliant = True

    async def teardown(self):
        return await super().teardown()


class TouchAntenna(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.touch_antenna = np.load('movements/touch_antenna.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
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
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        look_down = self.reachy.head.look_at_async(0.5, 0.8, 0.1, duration=1.0)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -180,
                self.reachy.head.l_antenna: 180,
            },
            duration=0.8,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        first_point = dict(zip(self.recorded_joints, self.touch_antenna[230]))
        # Goes to the start of the trajectory in 3s
        first_pos = goto_async(first_point, duration=1.2)

        await asyncio.gather(
            look_down,
            first_pos,
            traj_antennas,
        )

        for jp_arms in self.touch_antenna[230:580]:
            for joint, pos in zip(self.recorded_joints, jp_arms):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency))

        look_up = self.reachy.head.look_at_async(0.5, 0.0, 0.0, duration=1.0)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: 0.0,
                self.reachy.head.l_antenna: 0.0,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        last_point = dict(zip(self.recorded_joints, self.touch_antenna[780]))
        # Goes to the start of the trajectory in 3s
        last_pos = goto_async(last_point, duration=1.0)

        await asyncio.gather(
            look_up,
            last_pos,
            traj_antennas,
        )

    async def teardown(self):
        return await super().teardown()


class BeatRythmTable(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.beat_rythm_table = np.load('movements/beat_rythm_table.npy')
        self.beat_rythm_head = np.load('movements/beat_rythm_head.npy')

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

        self.recorded_head = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        first_point = dict(zip(self.recorded_joints, self.beat_rythm_table[0]))
        # Goes to the start of the trajectory in 1s
        first_pos = goto_async(first_point, duration=1.0)

        head_straight = self.reachy.head.look_at_async(0.5, 0, 0, duration=1.0)

        await asyncio.gather(
            head_straight,
            first_pos,
        )

        for (jp_head, jp_arms) in zip(self.beat_rythm_head[60:], self.beat_rythm_table):
            for joint, pos in zip(self.recorded_joints, jp_arms):
                joint.goal_position = pos
            for joint, pos in zip(self.recorded_head, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / self.sampling_frequency)
        
        await self.reachy.head.look_at_async(0.5, 0, 0, duration=1.5)
        

    async def teardown(self):
        return await super().teardown()


class StretchNeck(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.stretch_arm = np.load('movements/stretch_neck.npy')
        self.stretch_head = np.load('movements/stretch_neck_head.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            self.reachy.l_arm.l_shoulder_pitch,
            self.reachy.l_arm.l_shoulder_roll,
            self.reachy.l_arm.l_arm_yaw,
            self.reachy.l_arm.l_elbow_pitch,
            self.reachy.l_arm.l_forearm_yaw,
            self.reachy.l_arm.l_wrist_pitch,
            self.reachy.l_arm.l_wrist_roll,
        ]

        self.recorded_head = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        first_point = dict(zip(self.recorded_joints, self.stretch_arm[0]))
        # Goes to the start of the trajectory in 1s
        await goto_async(first_point, duration=1.0)

        for (jp_head, jp_arms) in zip(self.stretch_head, self.stretch_arm):
            for joint, pos in zip(self.recorded_joints, jp_arms):
                joint.goal_position = pos
            for joint, pos in zip(self.recorded_head, jp_head):
                joint.goal_position = pos

            await asyncio.sleep(1 / self.sampling_frequency)
        
        await self.reachy.head.look_at_async(0.5, 0, 0, duration=1.5)
        

    async def teardown(self):
        return await super().teardown()


class Sneeze(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.sneeze_sound = 'sounds/mixkit-little-baby-sneeze-2214.wav'


    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        await self.reachy.head.look_at_async(0.5, 0.0, 0.2, 0.8)
        await asyncio.sleep(0.7)
        playsound(self.sneeze_sound, block=False)
        await asyncio.sleep(0.3)
        head_pos1 = self.reachy.head.look_at_async(0.5, 0.0, -0.2, 0.2)
        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -30.0,
                self.reachy.head.l_antenna: 30.0,
            },
            duration=0.2,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        await asyncio.gather(
            head_pos1,
            traj_antennas,
        )
        head_pos2 = self.reachy.head.look_at_async(0.5, 0.0, 0.0, 1.0)
        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: 0.0,
                self.reachy.head.l_antenna: 0.0,
            },
            duration=0.5,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        await asyncio.gather(
            head_pos2,
            traj_antennas,
        )
        

    async def teardown(self):
        return await super().teardown()


class Whistle(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.whistle_sound = 'sounds/whistling.wav'

        self.head_movement = np.load('movements/whistling.npy')

        self.sampling_frequency = 100

        self.recorded_head = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        breathing = ArmBreathing(name = 'brt', reachy = self.reachy, fundamental_frequency=0.1)
        await breathing.start()

        await self.reachy.head.look_at_async(0.5, 0.0, 0.0, 1.0)

        first_point = dict(zip(self.recorded_head, self.head_movement[40]))
        # Goes to the start of the trajectory in 1s
        await goto_async(first_point, duration=0.5)
        
        for i in range(3):
            playsound(self.whistle_sound, block=False)

            for jp_head in self.head_movement[40:180]:
                for joint, pos in zip(self.recorded_head, jp_head):
                    joint.goal_position = pos

                await asyncio.sleep(1 / self.sampling_frequency)

        await breathing.stop()

    async def teardown(self):
        return await super().teardown()


class Surprise(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.huh_sound = 'sounds/huh_2.wav'


    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        look_huh =  self.reachy.head.look_at_async(0.5, 0.3, 0.05, 0.8)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: -60.0,
                self.reachy.head.l_antenna: 20.0,
            },
            duration=0.8,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        playsound(self.huh_sound, block=False)

        await asyncio.gather(
            look_huh,
            traj_antennas,
        )

        await asyncio.sleep(1)

        await goto_async(
            goal_positions={
                self.reachy.head.r_antenna: 0.0,
                self.reachy.head.l_antenna: 0.0,
            },
            duration=0.8,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

    async def teardown(self):
        return await super().teardown()


class Wonder(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.wow_sound = 'sounds/wow-1.wav'


    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        await self.reachy.head.look_at_async(0.5, -0.3, 0.0, 0.8)

        playsound(self.wow_sound, block=False)

        look_up = self.reachy.head.look_at_async(0.5, -0.3, 0.1, 0.2)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: 10.0,
                self.reachy.head.l_antenna: -15.0,
            },
            duration=0.2,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        await asyncio.gather(
            look_up,
            traj_antennas,
        )

        await asyncio.sleep(1)

        look_down = self.reachy.head.look_at_async(0.5, -0.3, -0.1, 0.2)

        traj_antennas = goto_async(
            goal_positions={
                self.reachy.head.r_antenna: 0.0,
                self.reachy.head.l_antenna: 0.0,
            },
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )

        await asyncio.gather(
            look_down,
            traj_antennas,
        )

    async def teardown(self):
        return await super().teardown()

class Sigh(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.sigh_sound = 'sounds/sigh-3.wav'


    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        playsound(self.sigh_sound, block=False)

        await self.reachy.head.look_at_async(0.5, 0.0, -0.3, 0.8)

        await asyncio.sleep(1.0)

        await self.reachy.head.look_at_async(0.5, 0.0, 0.0, 0.8)

    async def teardown(self):
        return await super().teardown()


class ClearThroat(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.clear_throat_sound = 'sounds/1191_clearing-throat-02.wav'


    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        playsound(self.clear_throat_sound, block=False)

    async def teardown(self):
        return await super().teardown()


class Cough(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.cough_sound = 'sounds/mixkit-strange-man-cough-2220.wav'


    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 0.0

        playsound(self.cough_sound, block=False)

    async def teardown(self):
        return await super().teardown()


class BlahBlah(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.blah_sound = 'sounds/blah-blah-blah.wav'

        self.arm_movement = np.load('movements/blahblah.npy')
        self.grip_movement = np.load('movements/blahblah_gripper.npy')
        self.head_movement = np.load('movements/blahblah_head.npy')

        self.sampling_frequency = 100

        self.gripper_joint = [reachy.l_arm.l_gripper]
        self.recorded_joints = [
            reachy.l_arm.l_shoulder_pitch,
            reachy.l_arm.l_shoulder_roll,
            reachy.l_arm.l_arm_yaw,
            reachy.l_arm.l_elbow_pitch,
            reachy.l_arm.l_forearm_yaw,
            reachy.l_arm.l_wrist_pitch,
            reachy.l_arm.l_wrist_roll,
        ]

        self.recorded_joints_head = [
            reachy.head.neck_roll,
            reachy.head.neck_pitch,
            reachy.head.neck_yaw,
        ]


    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        # Create a dict associating a joint to its first recorded position
        first_point = dict(zip(self.recorded_joints, self.arm_movement[150]))
        first_point_grip = dict(zip(self.gripper_joint, self.grip_movement[150]))
        first_point_head = dict(zip(self.recorded_joints_head, self.head_movement[150]))

        # Goes to the start of the trajectory in 3s
        arm_move = goto_async(first_point, duration=1.0)
        grip_move = goto_async(first_point_grip, duration=1.0)
        head_move = goto_async(first_point_head, duration=1.0)

        await asyncio.gather(
            arm_move,
            grip_move,
            head_move,
        )

        playsound(self.blah_sound, block=False)

        for (joints_positions, joints_positions_grip, joints_positions_head) in zip(self.arm_movement[150:], self.grip_movement[150:], self.head_movement[150:]):
            for joint, pos in zip(self.recorded_joints, joints_positions):
                joint.goal_position = pos
            for joint, pos in zip(self.gripper_joint, joints_positions_grip):
                joint.goal_position = pos
            for joint, pos in zip(self.recorded_joints_head, joints_positions_head):
                joint.goal_position = pos
            await asyncio.sleep(1 / (self.sampling_frequency*2))

    async def teardown(self):
        return await super().teardown()


class ArmBreathing(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False, fundamental_frequency: float = 0.3) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)
        self.fundamental_frequency = fundamental_frequency

    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        await goto_async(
            {self.reachy.l_arm.l_shoulder_pitch: 0.0,
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
            self.reachy.r_arm.r_gripper: 0.0,},
            duration=1.0,
        )

        dur = 20
        t = np.linspace(0, dur, dur * 100)

        t0 = time.time()

        while True:
            pos = 4 * np.sin(2 * np.pi * self.fundamental_frequency * (time.time()-t0))
            self.reachy.r_arm.r_arm_yaw.goal_position = pos
            self.reachy.l_arm.l_arm_yaw.goal_position = -pos
            pos2 = 1.5 * np.sin(2 * np.pi * self.fundamental_frequency * (time.time()-t0) + np.pi)
            self.reachy.r_arm.r_shoulder_roll.goal_position = pos2
            self.reachy.l_arm.l_shoulder_roll.goal_position = -pos2
            pos3 = 3 * np.sin(2 * np.pi * self.fundamental_frequency/2 * (time.time()-t0) + np.pi)
            self.reachy.r_arm.r_forearm_yaw.goal_position = -pos3
            self.reachy.l_arm.l_forearm_yaw.goal_position = -pos3
            pos4 = 4 * np.sin(2 * np.pi * self.fundamental_frequency * (time.time()-t0) + np.pi)
            self.reachy.r_arm.r_gripper.goal_position = -pos4
            self.reachy.l_arm.l_gripper.goal_position = pos4
            await asyncio.sleep(0.01)
        

    async def teardown(self):
        return await super().teardown()


class Idle(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)
        self.behaviors = {
            'look_at': LookAt(name='look_at', reachy=self.reachy, sub_behavior=True),
            'look_at2': LookAt2(name='look_at2', reachy=self.reachy, sub_behavior=True),
            'look_at3': LookAt3(name='look_at3', reachy=self.reachy, sub_behavior=True),
            'look_at4': LookAt4(name='look_at4', reachy=self.reachy, sub_behavior=True),
            'look_at5': LookAt5(name='look_at5', reachy=self.reachy, sub_behavior=True),
            'look_at6': LookAt6(name='look_at6', reachy=self.reachy, sub_behavior=True),
            'look_at7': LookAt7(name='look_at7', reachy=self.reachy, sub_behavior=True),
            'look_at8': LookAt8(name='look_at8', reachy=self.reachy, sub_behavior=True),
            # 'popote': Popote(name='popote', reachy=self.reachy, sub_behavior=True),
            'look_hand': LookHand(name='look_hand', reachy=self.reachy, sub_behavior=True),
            'lonely': Lonely(name='lonely', reachy=self.reachy, sub_behavior=True),
            'scratch' : Scratch(name='scratch', reachy=self.reachy, sub_behavior=True),
            'scratch2' : Scratch2(name='scratch2', reachy=self.reachy, sub_behavior=True),
            'scratch3' : Scratch3(name='scratch3', reachy=self.reachy, sub_behavior=True),
            'tshirt' : Tshirt(name='tshirt', reachy=self.reachy, sub_behavior=True),
            'sweat_head' : SweatHead(name='sweat_head', reachy=self.reachy, sub_behavior=True),
            'beat_rythm_table' : BeatRythmTable(name='beat_rythm_table', reachy=self.reachy, sub_behavior=True),
            'stretch_neck' : StretchNeck(name='stretch_neck', reachy=self.reachy, sub_behavior=True),
            'touch_antenna' : TouchAntenna(name='touch_antenna', reachy=self.reachy, sub_behavior=True),
            'sneeze' : Sneeze(name='sneeze', reachy=self.reachy, sub_behavior=True),
            'whistle' : Whistle(name='whistle', reachy=self.reachy, sub_behavior=True),
            'wonder' : Wonder(name='wonder', reachy=self.reachy, sub_behavior=True),
            'surprise' : Surprise(name='surprise', reachy=self.reachy, sub_behavior=True),
            'sigh' : Sigh(name='sigh', reachy=self.reachy, sub_behavior=True),
            'clear_throat' : ClearThroat(name='clear_throat', reachy=self.reachy, sub_behavior=True),
            'blahblah' : BlahBlah(name='blahblah', reachy=self.reachy, sub_behavior=True),
            'arm_breathing' : ArmBreathing(name='arm_breathing', reachy=self.reachy, sub_behavior=True),
        }

    async def run(self):
        while True:
            await self.behaviors[np.random.choice(list(self.behaviors.keys()))]._run()


class HeadController(Behavior):
    def __init__(self, name, reachy, sub_behavior, init_x=0.5, init_y=0.0, init_z=0.0):
        Behavior.__init__(self, name=name, reachy=reachy, sub_behavior=sub_behavior)
        self.x, self.y, self.z = init_x, init_y, init_z

    # async def setup(self):
    #     base_pos_right = [-1.73, -3.67, -0.57, -68.44, 4.0, -29.67, -4.84, -47.14]
    #     goto_dic = {j: pos for j, pos in zip(self.reachy.r_arm.joints.values(), base_pos_right)}
    #     arm_down = goto_async(goal_positions=goto_dic, duration=1.0)
    #     first_look_at = self.reachy.head.look_at_async(x=self.x, y=self.y, z=self.z, duration=1)
    #     await asyncio.gather(arm_down, first_look_at)

    #     for j in self.reachy.r_arm.joints.values():
    #         j.torque_limit = 0.0


    async def run(self):
        base_pos_right = [-1.73, -3.67, -0.57, -68.44, 4.0, -29.67, -4.84, -47.14]
        goto_dic = {j: pos for j, pos in zip(self.reachy.r_arm.joints.values(), base_pos_right)}
        arm_down = goto_async(goal_positions=goto_dic, duration=1.0)
        first_look_at = self.reachy.head.look_at_async(x=self.x, y=self.y, z=self.z, duration=1)
        await asyncio.gather(arm_down, first_look_at)

        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        while True:
            try:
                gp_dic = self.reachy.head._look_at(self.x, self.y, self.z)
                self.reachy.head.neck_disk_bottom.goal_position = gp_dic[self.reachy.head.neck_disk_bottom]
                self.reachy.head.neck_disk_middle.goal_position = gp_dic[self.reachy.head.neck_disk_middle]
                self.reachy.head.neck_disk_top.goal_position = gp_dic[self.reachy.head.neck_disk_top]
            except ValueError:
                pass
            await asyncio.sleep(0.01)


class FaceTracking(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy=reachy, sub_behavior=sub_behavior)

        self.detect = Detection(reachy=self.reachy, detection_model_path='/home/pierre/dev/reachy-face-tracking/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')
        self.idle = Idle(name='idle', reachy=reachy, sub_behavior=False)
        self.hc = HeadController(name='hc', reachy=reachy, sub_behavior=False)

        self.detect.start()

    async def run(self):
        center = np.array([160, 160])

        prev_y, prev_z = 0, 0
        cmd_y, cmd_z = prev_y, prev_z
        # xM, yM = 0, 0

        Kpy, Kpz, Kiy, Kiz, Kdy, Kdz = [0.00006, 0.00005, 0, 0, 0.017, 0.002]

        try:
            while True:
                if not self.detect.somebody_detected():
                    if not self.idle.is_running():
                        await self.idle.start()
                        await self.hc.stop()
                        print("idle")

                else:
                    x_target, y_target, _ = self.detect._face_target

                    target = np.array([y_target, x_target]) - center

                    cmd_z += np.round(-target[0] * Kpz, 3)
                    cmd_y += np.round(-target[1] * Kpy, 3)

                    self.hc.y = cmd_y
                    self.hc.z = cmd_z

                    if not self.hc.is_running():
                        await self.idle.stop()
                        await self.hc.start()
                        cmd_y, cmd_z = 0, 0
                        print("tracking")
                        print(f"cmd_y: {cmd_y}, cmd_z: {cmd_z}")

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            await self.idle.stop()
            await self.hc.stop()
            raise