"""The box while a run measures: the GPU's clock, power and temperature (nvidia-smi, every 2 s) and the
host's MemAvailable. Summary: the clock and power medians under load (GPU utilisation of 50 % and
up), the highest temperature, the lowest MemAvailable. Local reads only; nothing is sent anywhere."""

from __future__ import annotations

import statistics
import subprocess
import threading
import time

QUERY = "clocks.sm,power.draw,temperature.gpu,utilization.gpu"


def mem_available_gib() -> float | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        return None
    return None


def _num(x: str):
    try:
        return float(x)
    except ValueError:
        return None


class BoxSampler:
    def __init__(self, period: float = 2.0):
        self.period, self.samples, self.stop = period, [], threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.t0 = None

    def _loop(self):
        while not self.stop.is_set():
            s = {"t": round(time.monotonic() - self.t0, 1), "mem_avail_gib": mem_available_gib()}
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                parts = [_num(p.strip()) for p in out.splitlines()[0].split(",")] if out.strip() else []
                if len(parts) == 4:
                    s.update(clock_mhz=parts[0], power_w=parts[1], temp_c=parts[2], util=parts[3])
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
            self.samples.append(s)
            self.stop.wait(self.period)

    def __enter__(self):
        self.t0 = time.monotonic()
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.th.join(timeout=15)

    def summary(self) -> dict:
        load = [s for s in self.samples if (s.get("util") or 0) >= 50]

        def med(k, xs):
            v = [s[k] for s in xs if s.get(k) is not None]
            return round(statistics.median(v), 1) if v else None
        temps = [s["temp_c"] for s in self.samples if s.get("temp_c") is not None]
        mem = [s["mem_avail_gib"] for s in self.samples if s.get("mem_avail_gib") is not None]
        return {"samples": len(self.samples), "load_samples": len(load),
                "clock_mhz_median_load": med("clock_mhz", load), "power_w_median_load": med("power_w", load),
                "temp_c_max": max(temps) if temps else None, "mem_avail_gib_min": min(mem) if mem else None,
                "minutes": round((self.samples[-1]["t"] if self.samples else 0) / 60, 1)}
