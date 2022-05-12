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