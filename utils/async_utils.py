###############################################################################
#  Async utility helpers
###############################################################################

import asyncio
from typing import AsyncIterator, TypeVar

T = TypeVar("T")


async def merge_async_iters(*iterators: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    Merge multiple async iterators, yielding items in arrival order.
    Exits when all iterators are exhausted.
    """
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    active = len(iterators)

    async def _drain(it: AsyncIterator[T]):
        nonlocal active
        try:
            async for item in it:
                await queue.put(item)
        finally:
            active -= 1
            if active == 0:
                await queue.put(sentinel)

    tasks = [asyncio.create_task(_drain(it)) for it in iterators]
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
    finally:
        for task in tasks:
            task.cancel()


async def async_queue_iter(q: asyncio.Queue, sentinel=None) -> AsyncIterator:
    """Wrap an asyncio.Queue as an async iterator; exits when sentinel is received."""
    while True:
        item = await q.get()
        if item is sentinel:
            break
        yield item
