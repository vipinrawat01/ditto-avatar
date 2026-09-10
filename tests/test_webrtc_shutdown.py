import queue
import time

from server.webrtc import HumanPlayer, _put_latest


class _SlowThread:
    def join(self):
        time.sleep(0.5)


def test_player_stop_does_not_block_on_worker_shutdown():
    player = object.__new__(HumanPlayer)
    track = object()
    player._HumanPlayer__started = {track}
    player._HumanPlayer__thread_quit = type("Quit", (), {"set": lambda self: None})()
    player._HumanPlayer__thread = _SlowThread()
    player._HumanPlayer__container = object()

    started = time.perf_counter()
    player._stop(track)

    assert time.perf_counter() - started < 0.1


def test_media_queue_drops_oldest_instead_of_blocking():
    media_queue = queue.Queue(maxsize=2)
    media_queue.put_nowait("oldest")
    media_queue.put_nowait("newer")

    assert _put_latest(media_queue, "latest") is True
    assert media_queue.get_nowait() == "newer"
    assert media_queue.get_nowait() == "latest"
