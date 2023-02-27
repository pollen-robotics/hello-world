"""
Behavior class.

Define standard behaviors conception.
Behaviors can be cancelled when running.
"""
import asyncio


class Behavior:
    """Behavior class."""

    def __init__(self, name: str, reachy, sub_behavior: bool = False) -> None:
        """Intialize the behavior."""
        self.name = name
        self._task = None
        self.sub_behavior = sub_behavior
        self.reachy = reachy

    async def start(self):
        """Create asynchronous task used tu run the behavior."""
        await self.setup()
        self._task = asyncio.create_task(self._run(), name=f'behavior_{self.name}')
        return self._task

    async def stop(self):
        """Cancel the behavior."""
        if self._task is not None:
            self._task.cancel()
            await self._task

    async def setup(self):
        """Define setup method."""
        pass

    async def run(self):
        """Run method."""
        pass

    async def _run(self):
        try:
            await self.run()
        except asyncio.CancelledError:
            if self.sub_behavior:
                raise
        await self.teardown()

    async def teardown(self):
        """Define teardown method."""
        pass

    def is_running(self):
        """Return if the behavior is currently running."""
        return self._task is not None and not self._task.done()
