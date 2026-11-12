"""Cyberattacks injected between the sensors and the controller (and on the command path).

FDIA  : falsifies the SoC or load readings the controller receives.
DoS   : commands do not reach the field; the PV inverter falls back to a reduced output.
load  : the attacker switches on extra load at one bus (a physical change, seen by meters).
delay : the controller receives telemetry that is `intensity` minutes old.

An attack stops at its scheduled end, or earlier once it is detected and recovered:
end = onset + min(duration, detection latency + recovery time).
"""
import bisect
from dataclasses import dataclass
import config as C


@dataclass
class Attack:
    kind: str
    onset_min: float
    duration_min: float
    intensity: float
    channel: str = "all"          # fdia: 'soc' | 'load'; load: a group name; others: 'all'
    latency_min: float = 1e9      # detection latency (no detection by default)
    recovery_min: float = C.RECOVERY_MIN

    @property
    def end_min(self):
        return self.onset_min + min(self.duration_min, self.latency_min + self.recovery_min)

    def active(self, t):
        return self.onset_min <= t < self.end_min


def tamper(meas, t, attack, hist_t, hist_m):
    """Return the telemetry the controller receives at time t. hist_t/hist_m hold past true readings."""
    if attack is None or not attack.active(t):
        return meas
    if attack.kind == "fdia":
        if attack.channel == "soc":
            return dict(meas, soc=min(1.0, meas["soc"] + attack.intensity))
        return dict(meas, loads={g: v * (1 - attack.intensity) for g, v in meas["loads"].items()})
    if attack.kind == "delay":
        i = bisect.bisect_right(hist_t, t - attack.intensity) - 1
        return hist_m[max(i, 0)]
    return meas


def commands_blocked(t, attack):
    return attack is not None and attack.kind == "dos" and attack.active(t)


def extra_load(t, attack):
    """Extra kW per group switched on by a load-manipulation attack."""
    if attack is not None and attack.kind == "load" and attack.active(t):
        return {attack.channel: attack.intensity}
    return {}
