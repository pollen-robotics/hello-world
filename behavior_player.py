import asyncio

import reachy_sdk
import behavior

from reachy_sdk import ReachySDK


async def main():
    reachy = ReachySDK(host='localhost')
    reachy.turn_on('reachy')

    idle = behavior.Idle(name='idle', reachy=reachy)
    look_at = behavior.LookAt(name='look_at', reachy=reachy)
    look_hand = behavior.LookHand(name='look_hand', reachy=reachy)
    face_tracking = behavior.FaceTracking(name='face_tracking', reachy=reachy)
    lonely = behavior.Lonely(name='lonely', reachy=reachy)
    scratch = behavior.Scratch(name='scratch', reachy=reachy)

    t = await idle.start()
    await t


if __name__ == '__main__':
    asyncio.run(main())
