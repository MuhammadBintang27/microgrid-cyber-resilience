"""Microgrid controller: reads telemetry and decides load shedding and the PV power limit.

The battery itself is not dispatched: as the grid-forming source it balances the network
automatically. The controller protects it by shedding loads in order of criticality
(houses -> school -> pump -> clinic) and by limiting PV when the battery is nearly full.
"""
import numpy as np
import config as C


class Controller:
    def __init__(self):
        self.breakers = {g: True for g in C.GROUPS}
        self.hold_until = {g: -1.0 for g in C.GROUPS}   # groups shed for overload stay off a while

    def step(self, m, t):
        """m: telemetry {'soc', 'pv', 'loads': {group: kW}}. Returns (breaker commands, PV limit kW)."""
        soc = m["soc"]
        # 1. state-of-charge ladder with hysteresis
        for g in C.GROUPS:
            thr = C.LOADS[g]["shed_soc"]
            if soc < thr:
                self.breakers[g] = False
            elif soc > thr + C.RECONNECT_MARGIN and t >= self.hold_until[g]:
                self.breakers[g] = True
        # 2. overload guard: shed least critical groups until the inverter can carry the net load
        for g in C.SHED_ORDER:
            net = sum(m["loads"][x] for x in C.GROUPS if self.breakers[x]) - m["pv"]
            if net <= C.OVERLOAD_FRACTION * C.BAT_KW:
                break
            if self.breakers[g]:
                self.breakers[g] = False
                self.hold_until[g] = t + C.RESTART_DELAY_MIN
        # 3. PV limit: when the battery is nearly full, only produce what the loads use
        pv_limit = sum(m["loads"].values()) if soc >= C.CURTAIL_SOC else np.inf
        return dict(self.breakers), pv_limit
