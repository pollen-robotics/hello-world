import asyncio

import reachy_sdk
import behavior

from reachy_sdk import ReachySDK


async def main():
    reachy = ReachySDK(host='localhost')
    reachy.turn_on('reachy')

    idle = behavior.Idle(name='idle', reachy=reachy)
    look_at = behavior.LookAt(name='look_at', reachy=reachy)
    popote = behavior.Popote(name='popote', reachy=reachy)
    look_hand = behavior.LookHand(name='look_hand', reachy=reachy)
    face_tracking = behavior.FaceTracking(name='face_tracking', reachy=reachy)
    lonely = behavior.Lonely(name='lonely', reachy=reachy)
    scratch = behavior.Scratch(name='scratch', reachy=reachy)
    scratch2 = behavior.Scratch2(name='scratch2', reachy=reachy)
    tshirt = behavior.Tshirt(name='tshirt', reachy=reachy)
    sweat_head = behavior.SweatHead(name='sweat_head', reachy=reachy)
    beat_rythm_table = behavior.BeatRythmTable(name='beat_rythm_table', reachy=reachy)
    stretch_neck = behavior.StretchNeck(name='stretch_neck', reachy=reachy)


    t = await idle.start()
    await t


if __name__ == '__main__':
    asyncio.run(main())
