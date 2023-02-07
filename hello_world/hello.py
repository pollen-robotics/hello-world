"""
Helloworld application for Reachy2021.

This package defines an idle mode for Reachy, in which defined behaviors can be played randomly.
"""

import asyncio
import logging

from reachy_sdk import ReachySDK

from .behaviors.idle import Idle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == '__main__':
    reachy = ReachySDK(host='localhost')
    logger.info('Connected to Reachy')

    idle = Idle(name='idle', reachy=reachy)

    async def behavior():
        idle_behav = await idle.start()
        await idle_behav

    asyncio.run(behavior())
