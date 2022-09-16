# hello-world

The helloworld project defines an idle mode for Reachy.  
This project has been thought to:
* be able to make Reachy move immediately after the robot setup, without having to do any code
* start to elaborate a way to define behaviors for human-robot interaction, making independant components that can be reused and integrated in more complex programs. The idea is to be able to integrate this idle mode in any project requiring such a state, without having to re-code this behavior again and again.

## Discover helloworld

### Install the project

Clone the repository or download and extract the source code.  
Then install the project by going in the repo and doing: `pip3 install -e .`  

### Try the Idle mode

Make sure your robot is in the correct position: it needs to have space around it and the arms straight towards the floor, with no table under it.

Try it finally with the following command:  
`bash launch.bash`

### Project organization

The project is organized as following:
* **hello-world/behaviors**: contains the defined idle behaviors
* **movements**: contains the .npy of movements recorded that are called in some behaviors
* **sounds**: contains sounds to be play in some behaviors

## Add new behaviors

Each new behavior should inherite from the Behavior class. 
It should as well define at least the __init__, calling the super().__init__ inherited from the Behavior class, an async run function and an async teardown, calling the inheritied super().teardown function of Behavior.
A prototype is shown below:

```
class NewBehavior(Behavior):
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        super().__init__(name, reachy, sub_behavior=sub_behavior)

        # Initialize any other element relevant for your behavior


    async def run(self):
        for j in self.reachy.r_arm.joints.values():
            j.torque_limit = 100.0
        for j in self.reachy.l_arm.joints.values():
            j.torque_limit = 100.0

        # Your behavior code here

        self.reachy.turn_off_smoothly('r_arm')
        self.reachy.turn_off_smoothly('l_arm')

    async def teardown(self):
        return await super().teardown()
```

In order to have your behavior called in the idle function, you should then add an entry for your function in the behaviors dictionary of the Idle class:
in `hello_world/behaviors/idle.py`, setting sub_behavior to True:  
```
class Idle(Behavior):
    """Idle class."""

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
            'hello': Hello(name='hello', reachy=self.reachy, sub_behavior=True),
            # Add your new behavior here :
            # 'NEW_BEHAVIOR': NewBehavior(name='new_behavior', reachy=self.reachy, sub_behavior=True),
        }
```