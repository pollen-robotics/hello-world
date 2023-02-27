"""
In this script, we show how to play on Reachy one of the behavior developed for the hello-world
application. The goal is to easily visualize the behavior without having to run the application,
which plays behaviors randomly.
This script might also be used to test behaviors that you have implemented yourself.
"""
import asyncio
from grpc._channel import _InactiveRpcError
from reachy_sdk import ReachySDK
from .behaviors.asleep import Asleep
from .behaviors.look import LookHand
from .behaviors.moods import Lonely, Tshirt, SweatHead, Sneeze, Whistle, Hello
from .behaviors.scratch import Scratch


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'behavior',
        help="Reachy's recorded behavior that you want to play.",
        choices=[
            'asleep',
            'look_hand',
            'lonely',
            'scratch',
            'tshirt',
            'sweat_head',
            'sneeze',
            'whistle',
            'hello',
        ]
    )
    parser.add_argument('--ip_address', help="Reachy's ip address, default is 'localhost'.", default='localhost')

    args = parser.parse_args()
    requested_behavior = args.behavior
    reachy_ip_address = args.ip_address

    try:
        reachy = ReachySDK(host=reachy_ip_address)
    except _InactiveRpcError:
        print('Could not connect to Reachy. \n \
Make sure that reachy_sdk_server.service is running and that you entered the correct IP address.')
        exit()

    behavior_arg_to_class = {
        'asleep': Asleep(name='asleep', reachy=reachy),
        'look_hand': LookHand(name='look_hand', reachy=reachy),
        'lonely': Lonely(name='lonely', reachy=reachy),
        'scratch': Scratch(name='scratch', reachy=reachy),
        'tshirt': Tshirt(name='tshirt', reachy=reachy),
        'sweat_head': SweatHead(name='sweat_head', reachy=reachy),
        'sneeze': Sneeze(name='sneeze', reachy=reachy),
        'whistle': Whistle(name='whistle', reachy=reachy),
        'hello': Hello(name='hello', reachy=reachy),
    }

    # Make sure that the torque are correctly set at 100, in case
    # the previous turn_off_smoothly did not finish properly
    for joint in reachy.joints.values():
        joint.torque_limit = 100

    reachy.turn_on('reachy')

    async def behavior():
        print(f'Playing {requested_behavior} behavior.')
        await behavior_arg_to_class[requested_behavior].run()

    try:
        asyncio.run(behavior())

        print("Done! Turning off Reachy's motors.")

        # If the behavior played did not put the joints in stiff mode when finishing
        # Do it manually
        if not reachy.l_arm.l_shoulder_pitch.compliant:
            reachy.turn_off_smoothly('l_arm')
        if not reachy.r_arm.r_shoulder_pitch.compliant:
            reachy.turn_off_smoothly('r_arm')

    except KeyboardInterrupt:
        print(f'Ctrl-C received, stopping the {requested_behavior} behavior...')


if __name__ == '__main__':
    main()
