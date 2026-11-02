import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
from xgboost import XGBClassifier

# ---------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------
meta = joblib.load("pm_meta.joblib")
model = XGBClassifier()
model.load_model("pm_xgb.json")
type_model = meta["type_model"]
FEATURES = meta["features"]
TYPES = meta["types"]
THRESHOLD = meta["threshold"]
data = pd.read_csv("pm_clean.csv")

TYPE_NAMES = {
    "TWF": "Tool wear",
    "HDF": "Heat dissipation",
    "PWF": "Power",
    "OSF": "Overstrain",
}
TYPE_MAP = {"L": 0, "M": 1, "H": 2}
CV_RECALL, CV_PRECISION = 0.847, 0.732       # out-of-fold numbers from 5-fold CV

# Example costs used across the app (rupees)
COST_UNPLANNED, COST_PLANNED, COST_INSPECT = 200000, 40000, 5000

# Preset name -> [quality type, air K, process K, rpm, torque Nm, tool wear min]
PRESETS = {
    "Running normally":   ["M", 298.0, 308.6, 1500, 40.0, 30],
    "Cooling problem":    ["M", 302.0, 310.2, 1300, 45.0, 50],
    "Overloaded":         ["L", 298.0, 308.5, 1400, 70.0, 80],
    "Heavy strain":       ["L", 298.0, 308.6, 1450, 55.0, 230],
    "Worn-out tool":      ["M", 298.0, 308.6, 1500, 40.0, 235],
}

STRAIN_LIMIT = {"L": 11000, "M": 12000, "H": 13000}

# What a technician should do for each kind of failure
ACTIONS = {
    "TWF": ("Replace the cutting tool",
            "Swap the tool at the next planned stop and reset the wear counter.",
            "Cutting tool or insert"),
    "HDF": ("Inspect the cooling system",
            "Check coolant flow and the fan, and clear any blocked heat-exchanger passages.",
            "Coolant pump, fan, heat exchanger"),
    "PWF": ("Check motor load and power supply",
            "Reduce the feed rate, then inspect the motor, the drive and the supply.",
            "Motor, drive, supply cables"),
    "OSF": ("Reduce the load and inspect the tool holder",
            "Lower the cutting load and check the clamping and the tool condition.",
            "Tool holder, clamps"),
}

# Plot colours that read well on both light and dark pages
RED, GREEN, AMBER, GREY = "#ef4444", "#22c55e", "#f59e0b", "rgba(148,163,184,0.8)"
GRID, TEXT = "rgba(148,163,184,0.25)", "#94a3b8"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def build_features(type_code, air, proc, rpm, torque, wear):
    row = {
        "Type": type_code,
        "Air_temperature": air,
        "Process_temperature": proc,
        "Rotational_speed": rpm,
        "Torque": torque,
        "Tool_wear": wear,
        "Temp_diff": proc - air,
        "Power_W": torque * rpm * 2 * np.pi / 60,
        "Strain": wear * torque,
    }
    return pd.DataFrame([row])[FEATURES]


def prob_of(X):
    return float(model.predict_proba(X)[:, 1][0])


def type_probs(X):
    out = {}
    for name, p in zip(TYPES, type_model.predict_proba(X)):
        out[name] = float(p[0, 1]) if p.shape[1] > 1 else 0.0
    return out


def type_share(tp):
    """The type model scores each failure type on its own, and rare events always score low.
    Comparing them as a share of the total says how sure the diagnosis is among the four types."""
    total = sum(tp.values())
    return {k: (v / total if total > 1e-9 else 0.0) for k, v in tp.items()}


def style(fig, height=300, **kw):
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    layout.update(kw)          # caller overrides (e.g. margin) replace the defaults
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def card(kind, title, body=""):
    return (f'<div class="card {kind}"><div class="card-title">{title}</div>'
            f'<div class="card-body">{body}</div></div>')


def gauge(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT},
            "bar": {"color": RED if p >= THRESHOLD else GREEN, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, THRESHOLD * 100], "color": "rgba(34,197,94,0.15)"},
                {"range": [THRESHOLD * 100, 100], "color": "rgba(239,68,68,0.18)"},
            ],
            "threshold": {"line": {"color": AMBER, "width": 4},
                          "thickness": 0.85, "value": THRESHOLD * 100},
        },
    ))
    return style(fig, height=250, margin=dict(l=25, r=25, t=20, b=0))


def shap_fig(X):
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv)
        if sv.ndim == 3:
            sv = sv[..., 1]
        sv = sv.reshape(-1)[:len(FEATURES)]
    except Exception:
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker_color=AMBER))
        return style(fig, height=340, title="Overall feature importance")
    order = np.argsort(np.abs(sv))
    fig = go.Figure(go.Bar(
        x=sv[order], y=[FEATURES[i].replace("_", " ") for i in order], orientation="h",
        marker_color=[RED if v > 0 else "#3b82f6" for v in sv[order]]))
    return style(fig, height=340,
                 title="What is pushing the risk (red = up, blue = down)")


def watch_notes(t, air, proc, rpm, torque, wear):
    """Plain-English checks based on the known failure conditions of the AI4I dataset."""
    notes = []
    diff = proc - air
    power = torque * rpm * 2 * np.pi / 60
    strain = wear * torque
    if diff < 8.6 and rpm < 1380:
        notes.append(f"**Heat is not escaping.** The process runs only {diff:.1f} K above "
                     f"the air while the spindle turns at {rpm:.0f} rpm.")
    if power < 3500:
        notes.append(f"**Power is too low** ({power:,.0f} W).")
    elif power > 9000:
        notes.append(f"**Power is too high** ({power:,.0f} W).")
    if strain > STRAIN_LIMIT[t]:
        notes.append(f"**Overstrain.** Tool wear x torque is {strain:,.0f}, above the "
                     f"{STRAIN_LIMIT[t]:,} limit for type {t}.")
    if wear >= 200:
        notes.append(f"**The tool is near the end of its life** ({wear:.0f} min of use).")
    if not notes:
        return "All readings are inside their normal operating ranges."
    return "\n\n".join(f"- {n}" for n in notes)


# ---------------------------------------------------------------
# Tab: Check a machine
# ---------------------------------------------------------------
def load_preset(name):
    return PRESETS[name]


def predict(t, air, proc, rpm, torque, wear):
    X = build_features(TYPE_MAP[t], air, proc, rpm, torque, wear)
    p = prob_of(X)
    if p >= THRESHOLD:
        tp = type_probs(X)
        top = max(tp, key=tp.get)
        share = type_share(tp)
        status = card(
            "alert", "Likely to fail. Plan maintenance.",
            f"Risk is {p*100:.0f}%, above the {THRESHOLD*100:.0f}% alert line. "
            f"Most likely cause: <b>{TYPE_NAMES[top].lower()}</b>.")
        cause = go.Figure(go.Bar(
            x=[TYPE_NAMES[k] for k in TYPES], y=[share[k] * 100 for k in TYPES],
            marker_color=[RED if k == top else "rgba(148,163,184,0.5)" for k in TYPES]))
        cause = style(cause, height=230, title="Which kind of failure",
                      yaxis_title="Share of diagnosis (%)")
    else:
        status = card("ok", "Looks healthy",
                      f"Risk is {p*100:.0f}%, below the {THRESHOLD*100:.0f}% alert line.")
        cause = go.Figure()
        cause.add_annotation(text="Nothing to flag", showarrow=False,
                             font=dict(size=16, color=TEXT))
        cause.update_xaxes(visible=False)
        cause.update_yaxes(visible=False)
        cause = style(cause, height=230)
    notes = "### What the sensors say\n" + watch_notes(t, air, proc, rpm, torque, wear)
    return gauge(p), status, cause, notes, shap_fig(X)


# ---------------------------------------------------------------
# Tab: Watch it degrade (single machine)
# ---------------------------------------------------------------
def make_scenario(name, n=70):
    s = np.linspace(0, 1, n)
    rng = np.random.default_rng(7)
    air = 298 + rng.normal(0, 0.1, n)
    rpm = 1500 + rng.normal(0, 6, n)
    torque = 40 + rng.normal(0, 0.6, n)
    wear = 20 + 40 * s
    proc = air + 10.2 + rng.normal(0, 0.05, n)
    if name == "Worn tool, rising load":
        wear = 20 + 225 * s
        torque = 40 + 18 * s + rng.normal(0, 0.6, n)
    elif name == "Cooling loss":
        rpm = 1500 - 230 * s + rng.normal(0, 6, n)
        proc = air + 10.5 - 2.6 * s + rng.normal(0, 0.05, n)
    elif name == "Power overload":
        torque = 40 + 36 * s + rng.normal(0, 0.6, n)
        rpm = 1500 - 150 * s + rng.normal(0, 6, n)
    return air, proc, rpm, torque, wear


def simulate(scenario):
    n = 70
    air, proc, rpm, torque, wear = make_scenario(scenario, n)
    raw, shown, first = [], [], None
    msg = card("ok", "Monitoring...", "Readings are streaming in.")
    for i in range(n):
        Xi = build_features(1, air[i], proc[i], rpm[i], torque[i], wear[i])
        raw.append(prob_of(Xi))
        ps = float(np.mean(raw[-5:]))      # 5-step moving average, so one noisy reading cannot flip the alert
        shown.append(ps)

        if ps >= THRESHOLD and first is None:
            first = i
            tp = type_probs(Xi)
            top = max(tp, key=tp.get)
            msg = card("alert", f"Alert at step {i + 1}",
                       f"Likely cause: <b>{TYPE_NAMES[top].lower()}</b>. "
                       "Maintenance should be scheduled now.")

        fig = go.Figure(go.Scatter(
            y=np.array(shown) * 100, mode="lines",
            line=dict(color=RED, width=3),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)"))
        fig.add_hline(y=THRESHOLD * 100, line_dash="dash", line_color=AMBER,
                      annotation_text="Alert line", annotation_font_color=AMBER)
        fig = style(fig, height=300, yaxis=dict(range=[0, 100], title="Failure risk (%)"),
                    xaxis=dict(range=[0, n], title="Time step"))

        kind = "alert" if ps >= THRESHOLD else "ok"
        readout = card(
            kind, f"Health score {(1 - ps) * 100:.0f}%",
            f"Risk {ps * 100:.0f}% &nbsp;|&nbsp; Torque {torque[i]:.1f} Nm &nbsp;|&nbsp; "
            f"Speed {rpm[i]:.0f} rpm &nbsp;|&nbsp; Tool wear {wear[i]:.0f} min &nbsp;|&nbsp; "
            f"Temp. gap {proc[i] - air[i]:.1f} K")
        yield fig, readout, msg
        time.sleep(0.06)

    if first is None:
        msg = card("ok", "No alert in this run", "The machine stayed inside safe limits.")
    else:
        msg = card("alert", f"First alert at step {first + 1} of {n}",
                   f"The team had {n - first - 1} steps of warning before the end of the run.")
    yield fig, readout, msg


# ---------------------------------------------------------------
# Tab: Fleet overview (10 machines, live status, auto work orders)
# ---------------------------------------------------------------
FLEET_STEPS = 90          # one step = one simulated operating hour
WATCH_AT = THRESHOLD * 0.4

FLEET = [
    dict(id="M-01", name="Milling A", q="M", kind="healthy",   onset=0,  seed=11),
    dict(id="M-02", name="Milling B", q="M", kind="tool_wear", onset=12, seed=12),
    dict(id="M-03", name="Lathe A",   q="L", kind="healthy",   onset=0,  seed=13),
    dict(id="M-04", name="Lathe B",   q="M", kind="cooling",   onset=20, seed=14),
    dict(id="M-05", name="Drill A",   q="M", kind="healthy",   onset=0,  seed=15),
    dict(id="M-06", name="Drill B",   q="M", kind="power",     onset=8,  seed=16),
    dict(id="M-07", name="Press A",   q="L", kind="tool_wear", onset=30, seed=17),
    dict(id="M-08", name="Press B",   q="H", kind="healthy",   onset=0,  seed=18),
    dict(id="M-09", name="Grinder A", q="M", kind="cooling",   onset=35, seed=19),
    dict(id="M-10", name="Grinder B", q="L", kind="healthy",   onset=0,  seed=20),
]


def machine_series(m, n=FLEET_STEPS):
    """Sensor readings for one machine over n hours. Faulty machines drift from their onset hour."""
    rng = np.random.default_rng(m["seed"])
    t = np.arange(n)
    onset = m["onset"]
    full_at = int(0.85 * n)                                   # fully degraded by 85% of the shift
    s = np.clip((t - onset) / max(full_at - onset, 1), 0, 1)  # 0 = fine, 1 = fully degraded
    air = 298 + rng.normal(0, 0.1, n)
    rpm0 = 1500 + int(rng.integers(-30, 31))
    torque0 = 40 + int(rng.integers(-3, 4))
    wear0 = 15 + int(rng.integers(0, 40))
    gap0 = 10.2 + rng.uniform(-0.1, 0.2)
    rpm = rpm0 + rng.normal(0, 6, n)
    torque = torque0 + rng.normal(0, 0.6, n)
    wear = wear0 + 40 * t / (n - 1)                  # normal ageing
    gap = gap0 + rng.normal(0, 0.05, n)
    kind = m["kind"]
    if kind == "tool_wear":
        wear = wear + (245 - wear0 - 40) * s
        torque = torque + 16 * s
    elif kind == "cooling":
        rpm = rpm - 230 * s
        gap = gap + 0.3 - 2.6 * s
    elif kind == "power":
        torque = torque + 36 * s
        rpm = rpm - 150 * s
    torque = np.clip(torque, 3, 76.5)
    return air, air + gap, rpm, torque, wear


def fleet_frame(m):
    air, proc, rpm, torque, wear = machine_series(m)
    df = pd.DataFrame({
        "Type": TYPE_MAP[m["q"]],
        "Air_temperature": air,
        "Process_temperature": proc,
        "Rotational_speed": rpm,
        "Torque": torque,
        "Tool_wear": wear,
    })
    df["Temp_diff"] = df["Process_temperature"] - df["Air_temperature"]
    df["Power_W"] = df["Torque"] * df["Rotational_speed"] * 2 * np.pi / 60
    df["Strain"] = df["Tool_wear"] * df["Torque"]
    return df[FEATURES]


_FLEET_CACHE = {}


def fleet_precompute():
    """Score every machine at every hour in two batch calls, then just replay the results."""
    if _FLEET_CACHE:
        return _FLEET_CACHE
    nm, n = len(FLEET), FLEET_STEPS
    Xall = pd.concat([fleet_frame(m) for m in FLEET], ignore_index=True)
    p = model.predict_proba(Xall)[:, 1].reshape(nm, n)
    tp = np.column_stack([a[:, 1] if a.shape[1] > 1 else np.zeros(len(a))
                          for a in type_model.predict_proba(Xall)])
    _FLEET_CACHE["risk"] = np.vstack(
        [pd.Series(r).rolling(5, min_periods=1).mean().values for r in p])   # smoothed
    _FLEET_CACHE["type"] = tp.reshape(nm, n, len(TYPES))
    return _FLEET_CACHE


def spark(vals, color):
    w, h = 100, 26
    if len(vals) < 2:
        return ""
    last = FLEET_STEPS - 1
    pts = " ".join(f"{i / last * w:.1f},{h - 2 - min(v, 1) * (h - 4):.1f}"
                   for i, v in enumerate(vals))
    ty = h - 2 - THRESHOLD * (h - 4)
    return (f'<svg viewBox="0 0 {w} {h}" class="spark" preserveAspectRatio="none">'
            f'<line x1="0" x2="{w}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="{AMBER}" '
            f'stroke-dasharray="3 3" stroke-width="1" opacity="0.7" vector-effect="non-scaling-stroke"/>'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{pts}" '
            f'vector-effect="non-scaling-stroke"/></svg>')


def state_of(ps, alert_at):
    if alert_at is not None:
        return "alert"
    return "watch" if ps >= WATCH_AT else "ok"


def tile(m, vals, alert_at, cause):
    head = (f'<div class="tile-top"><span class="tile-name">{m["name"]}</span>'
            f'<span class="tile-id">{m["id"]} · type {m["q"]}</span></div>')
    if vals is None:
        return (f'<div class="tile idle">{head}<div class="tile-risk">--</div>'
                '<div class="bar"><div style="width:0%"></div></div>'
                '<div class="tile-foot">Waiting to start</div></div>')
    ps = float(vals[-1])
    kind = state_of(ps, alert_at)
    color = {"alert": RED, "watch": AMBER, "ok": GREEN}[kind]
    if kind == "alert":
        foot = f"<b>{TYPE_NAMES[cause]}</b> · alert since hour {alert_at + 1}"
    elif kind == "watch":
        foot = "Risk is climbing"
    else:
        foot = "Normal operation"
    health = max(0.0, 1 - ps) * 100
    return (f'<div class="tile {kind}">{head}'
            f'<div class="tile-risk">{ps * 100:.0f}%<span class="tile-sub"> risk</span></div>'
            f'<div class="bar"><div style="width:{health:.0f}%;background:{color}"></div></div>'
            f'{spark(vals, color)}<div class="tile-foot">{foot}</div></div>')


def render_fleet(t, alert_at, cause_at, risk, n_orders):
    tiles, counts = [], {"ok": 0, "watch": 0, "alert": 0}
    for i, m in enumerate(FLEET):
        vals = risk[i, :t + 1]
        counts[state_of(vals[-1], alert_at[i])] += 1
        tiles.append(tile(m, vals, alert_at[i], cause_at[i]))
    saved = n_orders * (COST_UNPLANNED - COST_PLANNED)
    summary = (
        '<div class="fleet-summary">'
        f'<span class="pill">Hour {t + 1} of {FLEET_STEPS}</span>'
        f'<span class="pill ok">{counts["ok"]} healthy</span>'
        f'<span class="pill watch">{counts["watch"]} watch</span>'
        f'<span class="pill alert">{counts["alert"]} alert</span>'
        f'<span class="pill">Up to ₹{saved:,} saved if every alert is acted on</span></div>')
    return summary + '<div class="fleet-grid">' + "".join(tiles) + "</div>"


def render_idle():
    summary = ('<div class="fleet-summary"><span class="pill">Shift not started</span>'
               f'<span class="pill">{len(FLEET)} machines connected</span></div>')
    return (summary + '<div class="fleet-grid">'
            + "".join(tile(m, None, None, None) for m in FLEET) + "</div>")


def work_order(number, m, step, cause, conf):
    lead = FLEET_STEPS - 1 - step
    priority = "Critical" if lead <= 15 else "High" if lead <= 35 else "Medium"
    action, detail, parts = ACTIONS[cause]
    return dict(
        order_id=f"WO-{1000 + number}",
        machine_id=m["id"], machine=m["name"], hour=step + 1,
        priority=priority, cause=TYPE_NAMES[cause], conf=conf * 100,
        action=action, detail=detail, parts=parts, lead=lead,
        saving=COST_UNPLANNED - COST_PLANNED)


def wo_html(o):
    prio = o["priority"].lower()
    return (f'<div class="wo {prio}">'
            f'<div class="wo-head"><span><b>{o["order_id"]}</b> &nbsp;{o["machine_id"]} · '
            f'{o["machine"]}</span><span class="chip {prio}">{o["priority"]}</span></div>'
            f'<div class="wo-body"><b>{o["action"]}.</b> {o["detail"]}<br>'
            f'Likely cause: {o["cause"].lower()} (diagnosis confidence {o["conf"]:.0f}%) · '
            f'raised at hour {o["hour"]} · about {o["lead"]} h of warning<br>'
            f'Parts to check: {o["parts"]}<br>'
            f'A planned fix costs ₹{COST_PLANNED:,}, an unplanned stop ₹{COST_UNPLANNED:,}. '
            f'That is up to <b>₹{o["saving"]:,}</b> saved.</div></div>')


def orders_html(orders):
    if not orders:
        return card("ok", "No work orders yet",
                    "The system raises one automatically when a machine crosses the alert line.")
    return "".join(wo_html(o) for o in reversed(orders))


def run_fleet(delay):
    R = fleet_precompute()
    risk, tprob = R["risk"], R["type"]
    alert_at = [None] * len(FLEET)
    cause_at = [None] * len(FLEET)
    orders = []
    for t in range(FLEET_STEPS):
        for i, m in enumerate(FLEET):
            if alert_at[i] is None and risk[i, t] >= THRESHOLD:
                k = int(np.argmax(tprob[i, t]))
                alert_at[i], cause_at[i] = t, TYPES[k]
                share = float(tprob[i, t, k] / max(tprob[i, t].sum(), 1e-9))
                orders.append(work_order(len(orders) + 1, m, t, TYPES[k], share))
        yield render_fleet(t, alert_at, cause_at, risk, len(orders)), orders_html(orders), None
        time.sleep(delay)

    path = None
    if orders:
        df = pd.DataFrame([{
            "order_id": o["order_id"], "machine_id": o["machine_id"], "machine": o["machine"],
            "raised_at_hour": o["hour"], "priority": o["priority"],
            "likely_cause": o["cause"], "diagnosis_confidence_pct": round(o["conf"]),
            "recommended_action": o["action"], "parts_to_check": o["parts"],
            "hours_of_warning": o["lead"], "planned_cost_inr": COST_PLANNED,
            "unplanned_cost_inr": COST_UNPLANNED, "potential_saving_inr": o["saving"],
        } for o in orders])
        path = os.path.join(tempfile.mkdtemp(), "work_orders.csv")
        df.to_csv(path, index=False)
    yield render_fleet(FLEET_STEPS - 1, alert_at, cause_at, risk, len(orders)), orders_html(orders), path


# ---------------------------------------------------------------
# Tab: ML vs a fixed-limit alarm
# ---------------------------------------------------------------
def baseline_compare(w_lim, t_lim, p_lim):
    d = data
    y = d["Machine_failure"].values.astype(bool)
    flag = ((d["Tool_wear"] >= w_lim) | (d["Torque"] >= t_lim)
            | (d["Process_temperature"] >= p_lim)).values
    n_fail = int(y.sum())

    tp_b = int((flag & y).sum())
    fp_b = int((flag & ~y).sum())
    fn_b = n_fail - tp_b
    rec_b = tp_b / max(n_fail, 1)
    prec_b = tp_b / max(tp_b + fp_b, 1)

    # ML counts are rebuilt from the cross-validated recall and precision
    tp_m = round(CV_RECALL * n_fail)
    alerts_m = round(tp_m / CV_PRECISION)
    fp_m = alerts_m - tp_m
    fn_m = n_fail - tp_m

    def cost(tp, fp, fn):
        return fn * COST_UNPLANNED + tp * COST_PLANNED + fp * COST_INSPECT

    cost_none = n_fail * COST_UNPLANNED
    cost_b, cost_m = cost(tp_b, fp_b, fn_b), cost(tp_m, fp_m, fn_m)

    c_fixed = card(
        "", "Fixed-limit alarm",
        f"Catches <b>{tp_b}</b> of {n_fail} failures ({rec_b*100:.0f}%)<br>"
        f"Misses <b>{fn_b}</b> · false alarms <b>{fp_b}</b><br>"
        f"Only {prec_b*100:.0f}% of its alerts are real<br>"
        f"Estimated cost: <b>₹{cost_b:,.0f}</b>")
    c_ml = card(
        "ok", "ML model (cross-validated)",
        f"Catches <b>{tp_m}</b> of {n_fail} failures ({CV_RECALL*100:.0f}%)<br>"
        f"Misses <b>{fn_m}</b> · false alarms <b>{fp_m}</b><br>"
        f"{CV_PRECISION*100:.0f}% of its alerts are real<br>"
        f"Estimated cost: <b>₹{cost_m:,.0f}</b>")
    gain = cost_b - cost_m
    if gain > 0:
        verdict = card("ok", f"The ML model costs ₹{gain:,.0f} less on this data",
                       f"With no monitoring at all, every failure is unplanned: "
                       f"₹{cost_none:,.0f}.")
    else:
        verdict = card("alert", "With these limits the fixed alarm is cheaper",
                       "Try tightening the limits and watch what happens to false alarms.")
    cards = f'<div class="two-col">{c_fixed}{c_ml}</div>{verdict}'

    f1 = go.Figure()
    f1.add_bar(name="Fixed-limit alarm", x=["Failures caught", "Alerts that were real"],
               y=[rec_b * 100, prec_b * 100], marker_color=GREY,
               text=[f"{rec_b*100:.0f}%", f"{prec_b*100:.0f}%"], textposition="outside")
    f1.add_bar(name="ML model", x=["Failures caught", "Alerts that were real"],
               y=[CV_RECALL * 100, CV_PRECISION * 100], marker_color=AMBER,
               text=[f"{CV_RECALL*100:.0f}%", f"{CV_PRECISION*100:.0f}%"], textposition="outside")
    f1 = style(f1, height=320, barmode="group", yaxis=dict(range=[0, 115], title="%"),
               legend=dict(orientation="h", y=1.15))

    names, vals = [], []
    for k in TYPES:
        if k in d.columns and d[k].sum() > 0:
            names.append(TYPE_NAMES[k])
            vals.append(float(flag[d[k].values.astype(bool)].mean() * 100))
    f2 = go.Figure(go.Bar(x=names, y=vals, marker_color=GREY,
                          text=[f"{v:.0f}%" for v in vals], textposition="outside"))
    f2.add_hline(y=CV_RECALL * 100, line_dash="dash", line_color=AMBER,
                 annotation_text="ML model, all failure types", annotation_font_color=AMBER)
    f2 = style(f2, height=320, title="Share of each failure type the fixed alarm catches",
               yaxis=dict(range=[0, 115], title="%"))
    return cards, f1, f2


def best_fixed_limits():
    """Cheapest combination of three single-sensor limits on all 10,000 readings.
    Tuning on the same data the alarm is scored on favours the alarm, so if the model
    still wins this is the hardest fair test a fixed-limit alarm can face."""
    y = data["Machine_failure"].values.astype(bool)
    wear = data["Tool_wear"].values
    torq = data["Torque"].values
    temp = data["Process_temperature"].values
    n_fail = int(y.sum())
    best = None
    for w in range(100, 255, 5):
        fw = wear >= w
        for t in range(40, 77, 2):
            ft = fw | (torq >= t)
            for p in np.arange(309, 314.01, 0.5):
                flag = ft | (temp >= p)
                tp = int((flag & y).sum())
                fp = int((flag & ~y).sum())
                c = (n_fail - tp) * COST_UNPLANNED + tp * COST_PLANNED + fp * COST_INSPECT
                if best is None or c < best[0]:
                    best = (c, w, t, round(float(p), 1))
    return best[1], best[2], best[3]


# ---------------------------------------------------------------
# Tab: What it saves
# ---------------------------------------------------------------
def impact(failures, c_unplanned, c_planned, c_inspect):
    failures = failures or 0
    caught = failures * CV_RECALL
    missed = failures - caught
    false_alarms = caught * (1 - CV_PRECISION) / CV_PRECISION
    without = failures * (c_unplanned or 0)
    with_ = missed * (c_unplanned or 0) + caught * (c_planned or 0) + false_alarms * (c_inspect or 0)

    fig = go.Figure(go.Bar(
        x=["Without the system", "With the system"], y=[without, with_],
        marker_color=[RED, GREEN], text=[f"₹{without:,.0f}", f"₹{with_:,.0f}"],
        textposition="outside"))
    fig = style(fig, height=320, yaxis_title="Yearly cost (₹)")

    saving = without - with_
    summary = card(
        "ok" if saving >= 0 else "alert",
        f"Estimated saving: ₹{saving:,.0f} per year",
        f"Caught early: <b>{caught:.0f}</b> &nbsp;|&nbsp; Missed: <b>{missed:.0f}</b> "
        f"&nbsp;|&nbsp; False alarms: <b>{false_alarms:.0f}</b>")
    note = (f"*Detection rates come from 5-fold cross-validation (recall {CV_RECALL*100:.1f}%, "
            f"precision {CV_PRECISION*100:.1f}%). The costs are example figures, so replace "
            "them with numbers from a real plant.*")
    return fig, summary, note


# ---------------------------------------------------------------
# Tab: Score a file
# ---------------------------------------------------------------
def score_csv(file):
    if file is None:
        return None, "Upload a CSV to begin."
    try:
        path = file if isinstance(file, str) else file.name
        df = pd.read_csv(path)
        Xb = pd.DataFrame({
            "Type": df["Type"].map(lambda v: TYPE_MAP.get(v, v)),
            "Air_temperature": df["Air_temperature"],
            "Process_temperature": df["Process_temperature"],
            "Rotational_speed": df["Rotational_speed"],
            "Torque": df["Torque"],
            "Tool_wear": df["Tool_wear"],
        })
        Xb["Temp_diff"] = Xb["Process_temperature"] - Xb["Air_temperature"]
        Xb["Power_W"] = Xb["Torque"] * Xb["Rotational_speed"] * 2 * np.pi / 60
        Xb["Strain"] = Xb["Tool_wear"] * Xb["Torque"]
        df["failure_risk_%"] = (model.predict_proba(Xb[FEATURES])[:, 1] * 100).round(1)
        df["alert"] = df["failure_risk_%"] >= THRESHOLD * 100
        df = df.sort_values("failure_risk_%", ascending=False)
        return df.head(200), f"**{int(df['alert'].sum())}** of {len(df)} readings raised an alert."
    except Exception as e:
        return None, f"Could not read that file: {e}"


# ---------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; }
.hero { padding: 22px 26px; border-radius: 16px; margin-bottom: 10px;
        background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(100,116,139,0.10));
        border: 1px solid rgba(245,158,11,0.35); }
.hero h1 { margin: 0 0 6px 0; font-size: 1.9rem; letter-spacing: -0.5px; }
.hero p { margin: 0; opacity: 0.8; font-size: 1.02rem; }
.card { border-radius: 12px; padding: 14px 18px; margin: 6px 0;
        border: 1px solid rgba(148,163,184,0.3); }
.card.ok { background: rgba(34,197,94,0.10); border-left: 5px solid #22c55e; }
.card.alert { background: rgba(239,68,68,0.10); border-left: 5px solid #ef4444; }
.card-title { font-weight: 700; font-size: 1.12rem; margin-bottom: 3px; }
.card-body { opacity: 0.88; line-height: 1.55; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.footer { text-align: center; opacity: 0.55; font-size: 0.85rem; margin-top: 18px; }

.fleet-summary { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 10px; }
.pill { padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 0.9rem;
        border: 1px solid rgba(148,163,184,0.35); }
.pill.ok { background: rgba(34,197,94,0.14); }
.pill.watch { background: rgba(245,158,11,0.18); }
.pill.alert { background: rgba(239,68,68,0.18); }
.fleet-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(205px, 1fr)); gap: 10px; }
.tile { border-radius: 12px; padding: 12px 14px; border: 1px solid rgba(148,163,184,0.3); }
.tile.ok { border-left: 5px solid #22c55e; background: rgba(34,197,94,0.07); }
.tile.watch { border-left: 5px solid #f59e0b; background: rgba(245,158,11,0.12); }
.tile.alert { border-left: 5px solid #ef4444; background: rgba(239,68,68,0.15); }
.tile.idle { border-left: 5px solid #64748b; background: rgba(100,116,139,0.08); }
.tile-top { display: flex; justify-content: space-between; align-items: baseline; gap: 6px; }
.tile-name { font-weight: 700; }
.tile-id { opacity: 0.6; font-size: 0.78rem; }
.tile-risk { font-size: 1.8rem; font-weight: 800; line-height: 1.15; }
.tile-sub { font-size: 0.8rem; font-weight: 500; opacity: 0.6; }
.bar { height: 6px; border-radius: 4px; background: rgba(148,163,184,0.25); overflow: hidden; margin: 6px 0; }
.bar > div { height: 100%; border-radius: 4px; }
.spark { width: 100%; height: 26px; display: block; }
.tile-foot { font-size: 0.82rem; opacity: 0.9; min-height: 1.3em; margin-top: 4px; }

.wo { border-radius: 12px; padding: 12px 16px; margin: 8px 0;
      border: 1px solid rgba(148,163,184,0.3); }
.wo.critical { border-left: 5px solid #ef4444; background: rgba(239,68,68,0.08); }
.wo.high { border-left: 5px solid #f59e0b; background: rgba(245,158,11,0.09); }
.wo.medium { border-left: 5px solid #3b82f6; background: rgba(59,130,246,0.08); }
.wo-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.wo-body { line-height: 1.6; opacity: 0.92; font-size: 0.95rem; }
.chip { padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
.chip.critical { background: rgba(239,68,68,0.25); }
.chip.high { background: rgba(245,158,11,0.28); }
.chip.medium { background: rgba(59,130,246,0.25); }
@media (max-width: 760px) { .two-col { grid-template-columns: 1fr; } }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

GR6 = int(gr.__version__.split(".")[0]) >= 6
blocks_kwargs = {} if GR6 else dict(theme=THEME, css=CSS)
launch_kwargs = dict(theme=THEME, css=CSS) if GR6 else {}

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
with gr.Blocks(title="Machine Health Monitor", **blocks_kwargs) as demo:
    gr.HTML(
        '<div class="hero"><h1>Machine Health Monitor</h1>'
        "<p>Reads the sensors on every machine on the floor and warns the maintenance team "
        "before a breakdown stops the line.</p></div>")

    # ---- Fleet overview --------------------------------------
    with gr.Tab("Fleet overview"):
        gr.Markdown(
            "Ten machines on one shop floor. Press start and watch them run: faults creep in "
            "on a few of them, and the system raises an alert and writes a work order "
            "**before** they fail. One step is one simulated operating hour.")
        with gr.Row():
            speed = gr.Slider(0.02, 0.4, value=0.08, step=0.01,
                              label="Seconds per simulated hour", scale=3)
            fleet_btn = gr.Button("Start the shift", variant="primary", scale=1)
        fleet_view = gr.HTML(value=render_idle())
        gr.Markdown("### Work orders")
        orders_view = gr.HTML(value=orders_html([]))
        csv_out = gr.File(label="Download the work orders (CSV)", interactive=False)
        fleet_btn.click(run_fleet, speed, [fleet_view, orders_view, csv_out])

    # ---- Check a machine -------------------------------------
    with gr.Tab("Check a machine"):
        preset = gr.Dropdown(list(PRESETS.keys()), value="Running normally",
                             label="Start from an example")
        with gr.Row():
            with gr.Column(scale=5):
                t = gr.Dropdown(["L", "M", "H"], value="M",
                                label="Product quality (L = low, M = medium, H = high)")
                air = gr.Slider(295, 305, value=298.0, step=0.1, label="Air temperature (K)")
                proc = gr.Slider(305, 314, value=308.6, step=0.1, label="Process temperature (K)")
                rpm = gr.Slider(1100, 2900, value=1500, step=10, label="Rotational speed (rpm)")
                torque = gr.Slider(3, 77, value=40, step=0.5, label="Torque (Nm)")
                wear = gr.Slider(0, 255, value=30, step=1, label="Tool wear (minutes in use)")
            with gr.Column(scale=6):
                g = gr.Plot(show_label=False)
                status = gr.HTML()
                notes = gr.Markdown()
        with gr.Row():
            cause = gr.Plot(show_label=False)
            why = gr.Plot(show_label=False)

        inputs = [t, air, proc, rpm, torque, wear]
        outputs = [g, status, cause, notes, why]

        # Presets fill the controls first, then the prediction runs once.
        preset.change(load_preset, preset, inputs).then(predict, inputs, outputs)
        # Sliders update on release, so the model is not called on every pixel of a drag.
        t.input(predict, inputs, outputs, show_progress="hidden")
        for s in (air, proc, rpm, torque, wear):
            s.release(predict, inputs, outputs, show_progress="hidden")
        demo.load(predict, inputs, outputs)

    # ---- Single machine simulation ---------------------------
    with gr.Tab("Watch it degrade"):
        gr.Markdown("Pick a fault and press start. One machine wears out step by step, "
                    "and the monitor should raise the alert **before** it fails.")
        with gr.Row():
            scen = gr.Dropdown(["Worn tool, rising load", "Cooling loss", "Power overload"],
                               value="Worn tool, rising load", label="Fault to simulate",
                               scale=3)
            go_btn = gr.Button("Start simulation", variant="primary", scale=1)
        sim_readout = gr.HTML()
        sim_plot = gr.Plot(show_label=False)
        sim_msg = gr.HTML()
        go_btn.click(simulate, scen, [sim_plot, sim_readout, sim_msg])

    # ---- ML vs fixed alarm -----------------------------------
    with gr.Tab("Why not a simple alarm?"):
        gr.Markdown(
            "Most plants today use fixed limits: ring the alarm when one sensor passes a "
            "number. Set those limits yourself below and see how they compare with the model "
            "on the same 10,000 readings.")
        with gr.Row():
            b_wear = gr.Slider(100, 250, value=200, step=5, label="Alarm when tool wear is at least (min)")
            b_torq = gr.Slider(40, 76, value=60, step=1, label="Alarm when torque is at least (Nm)")
            b_temp = gr.Slider(309, 314, value=311.5, step=0.1, label="Alarm when process temperature is at least (K)")
        b_best = gr.Button("Find the best fixed limits for me")
        b_cards = gr.HTML()
        with gr.Row():
            b_plot1 = gr.Plot(show_label=False)
            b_plot2 = gr.Plot(show_label=False)
        gr.Markdown(
            "*How to read this: the model's numbers come from 5-fold cross-validation, so it "
            "was scored on readings it had not seen. A fixed alarm has nothing to train, so it "
            "is scored on all 10,000 readings. The limits above are a plausible hand-made rule, "
            "not any real plant's settings. The button searches for the cheapest limits on the "
            "same data, which is the most generous test a fixed alarm can get. The data is the "
            "synthetic AI4I 2020 benchmark.*")
        b_in = [b_wear, b_torq, b_temp]
        b_out = [b_cards, b_plot1, b_plot2]
        for s in b_in:
            s.release(baseline_compare, b_in, b_out, show_progress="hidden")
        demo.load(baseline_compare, b_in, b_out)
        # Search for the cheapest fixed limits, move the sliders there, then re-score.
        b_best.click(best_fixed_limits, None, b_in).then(baseline_compare, b_in, b_out)

    # ---- Impact ----------------------------------------------
    with gr.Tab("What it saves"):
        gr.Markdown("Change the numbers to match your plant.")
        with gr.Row():
            f_in = gr.Number(value=100, label="Failures per year today")
            cu = gr.Number(value=COST_UNPLANNED, label="Cost of an unplanned failure (₹)")
            cp = gr.Number(value=COST_PLANNED, label="Cost of a planned repair (₹)")
            ci = gr.Number(value=COST_INSPECT, label="Cost of a false-alarm check (₹)")
        imp_txt = gr.HTML()
        imp_plot = gr.Plot(show_label=False)
        imp_note = gr.Markdown()
        imp_in = [f_in, cu, cp, ci]
        for comp in imp_in:
            comp.change(impact, imp_in, [imp_plot, imp_txt, imp_note], show_progress="hidden")
        demo.load(impact, imp_in, [imp_plot, imp_txt, imp_note])

    # ---- Batch -----------------------------------------------
    with gr.Tab("Score a file"):
        gr.Markdown("Upload a CSV with these columns: `Type`, `Air_temperature`, "
                    "`Process_temperature`, `Rotational_speed`, `Torque`, `Tool_wear`. "
                    "Type can be L/M/H or 0/1/2. Rows come back sorted by risk.")
        up = gr.File(label="CSV file", file_types=[".csv"])
        out_txt = gr.Markdown()
        out_df = gr.Dataframe()
        up.change(score_csv, up, [out_df, out_txt])

    gr.HTML('<div class="footer">Model: XGBoost trained on the public AI4I 2020 dataset '
            "(10,000 readings). It is a benchmark, not data from a real factory.</div>")

if __name__ == "__main__":
    # On Hugging Face Spaces the app must listen on all interfaces; locally keep it private.
    on_space = bool(os.environ.get("SPACE_ID"))
    demo.launch(server_name="0.0.0.0" if on_space else "127.0.0.1",
                server_port=7860, **launch_kwargs)