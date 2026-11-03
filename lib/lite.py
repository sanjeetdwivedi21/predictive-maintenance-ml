"""Pure-Python inference for the predictive-maintenance model.
No numpy, pandas, xgboost or scikit-learn needed, so it fits easily in a Vercel function."""
import json
import math
import os
import struct

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "model_lite.json")) as _f:
    M = json.load(_f)

FEATURES = M["features"]
TYPES = M["types"]
THRESHOLD = M["threshold"]
TYPE_NAMES = {"TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power", "OSF": "Overstrain"}
TYPE_MAP = {"L": 0, "M": 1, "H": 2}


def _f32(x):
    """scikit-learn casts inputs to float32 before comparing with a split threshold."""
    return struct.unpack("f", struct.pack("f", x))[0]


def build_row(quality, air, proc, rpm, torque, wear):
    return [float(TYPE_MAP[quality]), air, proc, rpm, torque, wear,
            proc - air, torque * rpm * 2 * math.pi / 60, wear * torque]


def failure_probability(x):
    m = M["base_margin"]
    for t in M["xgb"]:
        n, left, right, feat, val, dl = 0, t["l"], t["r"], t["f"], t["v"], t["d"]
        while left[n] != -1:
            xv = x[feat[n]]
            if xv != xv:                       # NaN follows the default direction
                n = left[n] if dl[n] else right[n]
            else:
                n = left[n] if xv < val[n] else right[n]
        m += val[n]
    return 1.0 / (1.0 + math.exp(-m))


def type_scores(x):
    """Score of each failure type on its own (one random forest per type)."""
    xf = [_f32(v) for v in x]
    out = {}
    for name, forest in M["rf"].items():
        total = 0.0
        for t in forest:
            n, left, right, feat, thr = 0, t["l"], t["r"], t["f"], t["t"]
            while left[n] != -1:
                n = left[n] if xf[feat[n]] <= thr[n] else right[n]
            total += t["p"][n]
        out[name] = total / len(forest)
    return out


def type_share(scores):
    total = sum(scores.values())
    return {k: (v / total if total > 1e-9 else 0.0) for k, v in scores.items()}


STRAIN_LIMIT = {"L": 11000, "M": 12000, "H": 13000}


def watch_notes(quality, air, proc, rpm, torque, wear):
    """Plain-English reading of the sensors, based on the known failure conditions of AI4I 2020."""
    notes = []
    gap = proc - air
    power = torque * rpm * 2 * math.pi / 60
    strain = wear * torque
    if gap < 8.6 and rpm < 1380:
        notes.append(f"Heat is not escaping: the process runs only {gap:.1f} K above the air "
                     f"while the spindle turns at {rpm:.0f} rpm.")
    if power < 3500:
        notes.append(f"Power is too low ({power:,.0f} W).")
    elif power > 9000:
        notes.append(f"Power is too high ({power:,.0f} W).")
    if strain > STRAIN_LIMIT[quality]:
        notes.append(f"Overstrain: tool wear x torque is {strain:,.0f}, above the "
                     f"{STRAIN_LIMIT[quality]:,} limit for type {quality}.")
    if wear >= 200:
        notes.append(f"The tool is near the end of its life ({wear:.0f} min of use).")
    return notes or ["All readings are inside their normal operating ranges."]
