"""
Idle mode definition.

Idle acts as a main behavior, and calls randomly defined behaviors as sub-behaviors.
Between each sub-behavior, awaits for the asleep behavior to be played.
"""
import logging
import numpy as np

from . import Behavior
from .asleep import Asleep
from .look import LookHand
from .moods import Lonely, Tshirt, SweatHead, Sneeze, Whistle, Hello
from .scratch import Scratch


class Idle(Behavior):
    """Idle class."""

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Initialize the behavior."""
        super().__init__(name, reachy=reachy, sub_behavior=sub_behavior)

        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger()

        self.reachy = reachy
        self.asleep_behavior = Asleep(name='asleep', reachy=self.reachy, sub_behavior=True)
        self.behaviors = {
            'look_hand': LookHand(name='look_hand', reachy=self.reachy, sub_behavior=True),
            'lonely': Lonely(name='lonely', reachy=self.reachy, sub_behavior=True),
            'scratch': Scratch(name='scratch', reachy=self.reachy, sub_behavior=True),
            'tshirt': Tshirt(name='tshirt', reachy=self.reachy, sub_behavior=True),
            'sweat_head': SweatHead(name='sweat_head', reachy=self.reachy, sub_behavior=True),
            'sneeze': Sneeze(name='sneeze', reachy=self.reachy, sub_behavior=True),
            'whistle': Whistle(name='whistle', reachy=self.reachy, sub_behavior=True),
            'hello': Hello(name='hello', reachy=self.reachy, sub_behavior=True)
        }

    async def run(self):
        """Implement the behavior."""
        while True:
            asleep = await self.asleep_behavior.start()
            self._logger.info('Playing asleep behavior.')
            await asleep
            self.reachy.turn_on('reachy')

            random_sub_behavior = np.random.choice(list(self.behaviors.keys()))
            self._logger.info(f'Playing sub behavior {random_sub_behavior}')
            await self.behaviors[random_sub_behavior]._run()

    async def teardown(self):
        """Put Reachy's motor in compliant mode when the Idle behavior stops."""
        self.reachy.turn_off_smoothly('reachy')
