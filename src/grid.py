"""Thin OpenDSS wrapper: load the model, set PV and load power, solve, read results."""
import numpy as np
import opendssdirect as dss
import config as C


def load_model():
    dss.Text.Command(f'Redirect "{C.DSS_FILE}"')


def solve(pv_kw, loads_kw):
    """Run one snapshot power flow. loads_kw: {group: kW}.

    Returns a flat dict:
      p_bat        power delivered by the battery inverter (kW, + = discharging)
      losses_kw    network losses (kW)
      v_<bus>_pu   mean voltage of the three phases
      i_<line>_pct largest phase current as % of the cable rating
    """
    dss.Generators.Name("pv")
    dss.Generators.kW(pv_kw)
    for name, kw in loads_kw.items():
        dss.Loads.Name(name)
        dss.Loads.kW(kw)
    dss.Solution.Solve()
    assert dss.Solution.Converged(), "power flow did not converge"

    # OpenDSS reports power delivered by a source as negative.
    dss.Circuit.SetActiveElement("Vsource.source")
    out = {"p_bat": -sum(dss.CktElement.Powers()[0:6:2]),
           "losses_kw": dss.Circuit.Losses()[0] / 1000}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        out[f"v_{bus}_pu"] = float(np.mean(dss.Bus.puVmagAngle()[0::2]))
    for line in dss.Lines.AllNames():
        dss.Lines.Name(line)
        out[f"i_{line}_pct"] = 100 * max(dss.CktElement.CurrentsMagAng()[0:6:2]) / dss.Lines.NormAmps()
    return out


def topology():
    """Bus coordinates {bus: (x, y)} and line ends {line: (bus1, bus2)} for drawings."""
    buses = {}
    for b in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(b)
        buses[b] = (dss.Bus.X(), dss.Bus.Y())
    lines = {}
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        lines[name] = (dss.Lines.Bus1().split(".")[0], dss.Lines.Bus2().split(".")[0])
    return buses, lines


def line_params(name):
    """Total (R, X) of one line in ohm, for the hand-calculation check."""
    dss.Lines.Name(name)
    return dss.Lines.R1() * dss.Lines.Length(), dss.Lines.X1() * dss.Lines.Length()
