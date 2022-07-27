"""
Helloworld application for Reachy2021.

This package defines an idle mode for Reachy, in which defined behaviors can be played randomly.
"""

import asyncio

from reachy_sdk import ReachySDK

from .behaviors.idle import Idle


if __name__ == '__main__':
    reachy = ReachySDK(host='localhost')
    idle = Idle(name='idle', reachy=reachy)

    async def behavior():
        idle_behav = await idle.start()
        await idle_behav

    asyncio.run(behavior())
