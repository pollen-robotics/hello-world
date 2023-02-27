"""
Moods behaviors.

Definiton of miscellaneous behaviors for Reachy.
"""

import asyncio
import numpy as np

from reachy_sdk.trajectory import goto_async
from reachy_sdk.trajectory.interpolation import InterpolationMode

from . import Behavior
from .player import playsound


class Lonely(Behavior):
    """
    Lonely class.

    Makes Reachy look around and act sad at the end.

    Uses: head
    Dependencies to other behaviors: none
    """

    async def run(self):
        """Implement the behavior."""
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
            starting_positions={
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
            starting_positions={
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

        await asyncio.gather(
            traj_antennas,
            first_look_down,
        )

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

        await asyncio.gather(
            traj_antennas,
            look_straight,
        )


class Tshirt(Behavior):
    """
    Tshirt class.

    Makes Reachy grab its T-shirt, and put it back in place.

    Uses: left_arm, head
    Dependencies to other behaviors: none
    """

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Initialize the Tshirt behavior."""
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
        """Implement the behavior."""
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
        first_pos = goto_async(first_point, duration=1.0)

        await asyncio.gather(
            look_down,
            first_pos,
        )

        for jp_arms in self.touch_tshirt[100:]:
            for joint, pos in zip(self.recorded_joints, jp_arms):
                joint.goal_position = pos

            await asyncio.sleep(1 / (self.sampling_frequency * 1.5))

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
            self.reachy.l_arm.l_shoulder_pitch: 0.0,
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

        await asyncio.sleep(0.3)

    async def teardown(self):
        """Use the teardown method from parent class."""
        return await super().teardown()


class SweatHead(Behavior):
    """
    SweatHead class.

    Makes Reachy touch wipe its head as if it was too hot.

    Uses: left_arm, head
    Dependencies to other behaviors: none
    """

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Initialize the SweatHead behavior."""
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
        """Implement the behavior."""
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

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

        last_pos = goto_async({
            self.reachy.l_arm.l_shoulder_pitch: 0.0,
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
        """Use the teardown method from parent class."""
        return await super().teardown()


class Hello(Behavior):
    """
    Hello class.

    Makes Reachy wave its hand to say hello.

    Uses: left_arm, head
    Dependencies to other behaviors: none
    """

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Initialize the Hello behavior."""
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.move_antennas = np.load('movements/hello_move_antennas.npy')
        self.move_arm = np.load('movements/hello_move.npy')

        self.sampling_frequency = 100

        self.recorded_joints_arm = [
            self.reachy.l_arm.l_shoulder_pitch,
            self.reachy.l_arm.l_shoulder_roll,
            self.reachy.l_arm.l_arm_yaw,
            self.reachy.l_arm.l_elbow_pitch,
            self.reachy.l_arm.l_forearm_yaw,
            self.reachy.l_arm.l_wrist_pitch,
            self.reachy.l_arm.l_wrist_roll,
        ]

        self.recorded_joints_antennas = [
            self.reachy.head.l_antenna,
            self.reachy.head.r_antenna,
        ]

    async def run(self):
        """Implement the behavior."""
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        head_move = goto_async(
            {
                self.reachy.head.neck_roll: 10.9,
                self.reachy.head.neck_pitch: 7.11,
                self.reachy.head.neck_yaw: 8.35,
            },
            duration=0.4,
        )

        first_point = dict(zip(self.recorded_joints_arm, self.move_arm[150]))
        arm_move = goto_async(first_point, duration=0.4)
        first_point_antennas = dict(zip(self.recorded_joints_antennas, self.move_antennas[50]))
        antennas_move = goto_async(first_point_antennas, duration=0.4)

        await asyncio.gather(
            head_move,
            arm_move,
            antennas_move
        )

        for (jp_antennas, jp_arms) in zip(self.move_antennas[50:], self.move_arm[150:500]):
            for joint, pos in zip(self.recorded_joints_arm, jp_arms):
                joint.goal_position = pos
            for joint, pos in zip(self.recorded_joints_antennas, jp_antennas):
                joint.goal_position = pos

            await asyncio.sleep(1 / self.sampling_frequency)

        last_pos = goto_async({
                self.reachy.l_arm.l_shoulder_pitch: 0.0,
                self.reachy.l_arm.l_shoulder_roll: 0.0,
                self.reachy.l_arm.l_arm_yaw: 0.0,
                self.reachy.l_arm.l_elbow_pitch: 0.0,
                self.reachy.l_arm.l_forearm_yaw: 0.0,
                self.reachy.l_arm.l_wrist_pitch: 0.0,
                self.reachy.l_arm.l_wrist_roll: 0.0},
                duration=1.0,
        )

        last_look_at = self.reachy.head.look_at_async(0.5, 0, 0, 0.5)

        await asyncio.gather(
            last_look_at,
            last_pos,
        )

        self.reachy.turn_off_smoothly('l_arm')

    async def teardown(self):
        """Use the teardown method from parent class."""
        return await super().teardown()


class Sneeze(Behavior):
    """
    Sneeze class.

    Makes Reachy sneeze with a head movement.

    Uses: head, sound
    Dependencies to other behaviors: none
    """

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Initialize the Sneeze behavior."""
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.sneeze_sound = 'sounds/sneezing.wav'

    async def run(self):
        """Implement the behavior."""
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
        """Use the teardown method from parent class."""
        return await super().teardown()


class Whistle(Behavior):
    """
    Whistle class.

    Makes Reachy whistle a tune cheerfully.

    Uses: right_arm, left_arm, head, sound
    Dependencies to other behaviors: ArmRythm
    """

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Initialize the Whistle behavior."""
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.whistle_sound = 'sounds/whistling.wav'

        self.head_movement = np.load('movements/whistling.npy')

        self.sampling_frequency = 100

        self.recorded_head = [
            self.reachy.head.neck_roll,
            self.reachy.head.neck_pitch,
            self.reachy.head.neck_yaw,
        ]

        self.recorded_joints = [
            reachy.l_arm.l_shoulder_pitch,
            reachy.l_arm.l_shoulder_roll,
            reachy.l_arm.l_arm_yaw,
            reachy.l_arm.l_elbow_pitch,
            reachy.l_arm.l_forearm_yaw,
            reachy.l_arm.l_wrist_pitch,
            reachy.l_arm.l_wrist_roll,
            reachy.r_arm.r_shoulder_pitch,
            reachy.r_arm.r_shoulder_roll,
            reachy.r_arm.r_arm_yaw,
            reachy.r_arm.r_elbow_pitch,
            reachy.r_arm.r_forearm_yaw,
            reachy.r_arm.r_wrist_pitch,
            reachy.r_arm.r_wrist_roll,
        ]

    async def run(self):
        """Implement the behavior."""
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 0.0

        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        arm_move = ArmRythm(name='arm_move', reachy=self.reachy)
        await arm_move.start()

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

        await arm_move.stop()

    async def teardown(self):
        """Use the teardown method from parent class."""
        return await super().teardown()


class ArmRythm(Behavior):
    """
    ArmRythm class.

    Makes Reachy moves its arms in rythm.

    Uses: right_arm, left_arm
    Dependencies to other behaviors: none
    """

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Initialize the ArmRythm behavior."""
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        self.arm_movement = np.load('movements/whistle_arms.npy')

        self.sampling_frequency = 100

        self.recorded_joints = [
            reachy.l_arm.l_shoulder_pitch,
            reachy.l_arm.l_shoulder_roll,
            reachy.l_arm.l_arm_yaw,
            reachy.l_arm.l_elbow_pitch,
            reachy.l_arm.l_forearm_yaw,
            reachy.l_arm.l_wrist_pitch,
            reachy.l_arm.l_wrist_roll,
            reachy.r_arm.r_shoulder_pitch,
            reachy.r_arm.r_shoulder_roll,
            reachy.r_arm.r_arm_yaw,
            reachy.r_arm.r_elbow_pitch,
            reachy.r_arm.r_forearm_yaw,
            reachy.r_arm.r_wrist_pitch,
            reachy.r_arm.r_wrist_roll,
        ]

    async def run(self):
        """Implement the behavior."""
        for jp_arm in self.arm_movement:
            for joint, pos in zip(self.recorded_joints, jp_arm):
                joint.goal_position = pos

            await asyncio.sleep(1 / self.sampling_frequency)

    async def teardown(self):
        """Use the teardown method from parent class."""
        return await super().teardown()
