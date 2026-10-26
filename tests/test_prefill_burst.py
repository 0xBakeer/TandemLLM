"""SPD-245: the prefill burst (server/prefill_burst.py): who joins, who waits, who closes."""

import threading
import time

from server.prefill_burst import Burst


def test_alone_never_waits():
    b = Burst(wait_s=5)
    assert b.arrive(1, idle=True)
    calls = []
    assert b.prefilled(1, release=lambda: calls.append("r"), acquire=lambda: calls.append("a")) is False
    assert calls == [] and not b.is_open()


def test_busy_board_opens_no_burst():
    b = Burst()
    assert not b.arrive(1, idle=False)
    assert b.prefilled(1) is False


def test_members_wait_for_the_last_prefill():
    b = Burst(wait_s=5)
    for i in range(4):
        assert b.arrive(i, idle=(i == 0))          # the later ones join the open burst
    order, lock = [], threading.Lock()

    def member(i):
        waited = b.prefilled(i, release=lambda: None, acquire=lambda: None)
        with lock:
            order.append((i, waited, time.monotonic()))

    th = [threading.Thread(target=member, args=(i,)) for i in range(3)]
    for t in th:
        t.start()
    time.sleep(0.2)
    assert not order                               # 0..2 wait for 3
    t_close = time.monotonic()
    member(3)
    for t in th:
        t.join(2)
    assert len(order) == 4
    assert all(w for i, w, _ in order if i != 3) and [w for i, w, _ in order if i == 3] == [False]
    assert all(t >= t_close for _, _, t in order)
    assert b.stats["bursts"] == 1 and b.stats["max_members"] == 4 and not b.is_open()


def test_turn_given_back_while_waiting():
    b = Burst(wait_s=5)
    b.arrive("a", idle=True)
    b.arrive("b", idle=True)
    log = []
    t = threading.Thread(target=lambda: b.prefilled("a", release=lambda: log.append("release"),
                                                   acquire=lambda: log.append("acquire")))
    t.start()
    time.sleep(0.1)
    assert log == ["release"]
    b.prefilled("b")
    t.join(2)
    assert log == ["release", "acquire"]


def test_leave_releases_the_waiters():
    b = Burst(wait_s=5)
    b.arrive(1, idle=True)
    b.arrive(2, idle=True)
    t = threading.Thread(target=lambda: b.prefilled(1))
    t.start()
    time.sleep(0.1)
    b.leave(2)                                     # 2 ended without its prefill
    t.join(2)
    assert not t.is_alive() and not b.is_open() and b.stats["timeouts"] == 0


def test_timeout_goes_on():
    b = Burst(wait_s=0.2)
    b.arrive(1, idle=True)
    b.arrive(2, idle=True)
    t0 = time.monotonic()
    assert b.prefilled(1) is True
    assert 0.15 < time.monotonic() - t0 < 2
    assert b.stats["timeouts"] == 1
    b.prefilled(2)                                 # the late one closes nothing and does not wait
    assert not b.is_open()


def test_a_new_burst_after_the_close():
    b = Burst(wait_s=5)
    b.arrive(1, idle=True)
    b.prefilled(1)
    assert not b.arrive(1, idle=False)             # decoding now: no burst
    assert b.arrive(2, idle=True) and b.member(2)


def test_join_runs_every_job_once_in_the_last_members_thread():
    b = Burst(wait_s=5)
    for i in range(4):
        b.arrive(i, idle=True)
    runs, got, lock = [], {}, threading.Lock()

    def run(jobs):
        runs.append((threading.current_thread().name, list(jobs)))
        return [j * 10 for j in jobs]

    def member(i):
        r = b.join(i, i + 1, run, release=lambda: None, acquire=lambda: None, rows=512)
        with lock:
            got[i] = r

    th = [threading.Thread(target=member, args=(i,), name=f"m{i}") for i in range(3)]
    for t in th:
        t.start()
    time.sleep(0.2)
    assert not runs and not got
    member(3)
    for t in th:
        t.join(2)
    assert len(runs) == 1 and runs[0][0] == threading.current_thread().name
    assert sorted(runs[0][1]) == [1, 2, 3, 4]
    assert got == {i: (True, (i + 1) * 10) for i in range(4)}
    assert b.stats["joint"] == 1 and b.stats["joint_rows"] == 2048 and not b.is_open()


def test_join_without_a_job_goes_alone_and_the_others_still_join():
    b = Burst(wait_s=5)
    for i in range(3):
        b.arrive(i, idle=True)
    got = {}
    t = threading.Thread(target=lambda: got.update({0: b.join(0, "a", lambda js: [j.upper() for j in js])}))
    t.start()
    time.sleep(0.1)
    assert b.join(1, None, None) == (False, None)  # a long prompt: alone, no wait
    assert b.join(2, "c", lambda js: [j.upper() for j in js]) == (True, "C")
    t.join(2)
    assert got[0] == (True, "A") and not b.is_open()


def test_join_of_one_runs_alone():
    b = Burst(wait_s=5)
    b.arrive(0, idle=True)
    called = []
    assert b.join(0, "x", lambda js: called.append(js)) == (False, None) and not called


def test_join_leader_error_reaches_every_member():
    b = Burst(wait_s=5)
    b.arrive(0, idle=True)
    b.arrive(1, idle=True)
    seen = {}

    def m0():
        try:
            b.join(0, 1, None)
        except RuntimeError as e:
            seen[0] = str(e)

    t = threading.Thread(target=m0)
    t.start()
    time.sleep(0.1)

    def boom(jobs):
        raise RuntimeError("oom")
    try:
        b.join(1, 2, boom)
    except RuntimeError as e:
        seen[1] = str(e)
    t.join(2)
    assert seen == {0: "oom", 1: "oom"} and not b.is_open()


def test_join_timeout_before_the_last_member_goes_alone():
    b = Burst(wait_s=0.2)
    b.arrive(0, idle=True)
    b.arrive(1, idle=True)
    assert b.join(0, "a", None) == (False, None)
    assert b.stats["timeouts"] == 1
    assert b.join(1, "b", lambda js: [j for j in js]) == (False, None)   # nothing left to join
    assert not b.is_open()


def test_leave_frees_waiting_joiners():
    b = Burst(wait_s=5)
    b.arrive(0, idle=True)
    b.arrive(1, idle=True)
    got = {}
    t = threading.Thread(target=lambda: got.update({0: b.join(0, "a", None)}))
    t.start()
    time.sleep(0.1)
    b.leave(1)
    t.join(2)
    assert got[0] == (False, None) and not b.is_open()


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
