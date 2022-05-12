import numpy as np

from . import Behavior


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