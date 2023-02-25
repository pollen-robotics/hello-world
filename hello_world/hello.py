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

    # Make sure that the torque are correctly set at 100, in case
    # the previous turn_off_smoothly did not finish properly
    for joint in reachy.joints.values():
        joint.torque_limit = 100

    idle = Idle(name='idle', reachy=reachy)

    async def behavior():
        idle_behav = await idle.start()
        await idle_behav

    try:
        asyncio.run(behavior())
    except KeyboardInterrupt:
        logger.info('Ctrl-C received, turning off the application...')
