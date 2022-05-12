import asyncio


class Behavior:
    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        self.name = name
        self._task = None
        self.sub_behavior = sub_behavior
        self.reachy = reachy

    async def start(self):
        await self.setup()
        self._task = asyncio.create_task(self._run(), name=f'behavior_{self.name}')
        return self._task

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            await self._task

    async def setup(self):
        pass

    async def run(self):
        pass

    async def _run(self):
        try:
            await self.run()
        except asyncio.CancelledError:
            if self.sub_behavior:
                raise
        await self.teardown()

    async def teardown(self):
        pass

    def is_running(self):
        return self._task is not None and not self._task.done()
