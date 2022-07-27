from threading import Thread

from playsound import playsound as _playsound


def playsound(sound, block):
    """Read sound without blocking and sound card saturation."""
    t = Thread(target=lambda: _playsound(sound, block=True))
    t.start()
    if block:
        t.join()
