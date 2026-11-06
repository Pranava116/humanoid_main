from robot.core.loop import ControlLoop


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, s):
        self.now += s


def test_virtual_time_runs_exact_ticks():
    loop = ControlLoop(50, realtime=False)
    ts = []
    stats = loop.run(lambda i: ts.append(i.t), duration=1.0)
    assert stats.ticks == 50
    assert ts[1] - ts[0] == 0.02


def test_realtime_schedule_with_fake_clock():
    ft = FakeTime()
    loop = ControlLoop(50, realtime=True, clock=ft.clock, sleep=ft.sleep)
    starts = []
    loop.run(lambda i: starts.append(ft.now), max_ticks=5)
    assert [round(s, 4) for s in starts] == [0.0, 0.02, 0.04, 0.06, 0.08]
    assert loop.stats.overruns == 0


def test_overrun_and_resync():
    ft = FakeTime()
    loop = ControlLoop(50, realtime=True, overrun_warn_ms=5, resync_after_ticks=2, clock=ft.clock, sleep=ft.sleep)

    def slow(i):
        if i.tick == 2:
            ft.now += 0.1     # 5 periods of work
    loop.run(slow, max_ticks=6)
    assert loop.stats.overruns >= 1
    assert loop.stats.skipped >= 3


def test_callback_false_stops():
    loop = ControlLoop(50, realtime=False)
    stats = loop.run(lambda i: i.tick < 3)
    assert stats.ticks == 4
