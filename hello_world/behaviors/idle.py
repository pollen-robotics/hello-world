import numpy as np

from . import Behavior
from .asleep import Asleep
from .look import LookHand
from .moods import Lonely, Tshirt, SweatHead, Sneeze, Whistle, Hello
from .scratch import Scratch


class Idle(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy=reachy, sub_behavior=sub_behavior)
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
        while True:
            asleep = await self.asleep_behavior.start()
            await asleep
            self.reachy.turn_on('reachy')
            await self.behaviors[np.random.choice(list(self.behaviors.keys()))]._run()

    async def teardown(self):
        self.reachy.turn_off_smoothly('reachy')
