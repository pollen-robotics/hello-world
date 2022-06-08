from threading import Thread

from playsound import playsound as _playsound


def playsound(sound, block):
    t = Thread(target=lambda: _playsound(sound, block=True))
    t.start()
    if block:
        t.join()
