import json
import os
import sys
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))
import lite as L

LIMITS = {"air_temperature": (250, 350), "process_temperature": (250, 350),
          "rotational_speed": (1, 10000), "torque": (0, 200), "tool_wear": (0, 1000)}


def validate(p):
    q = p.get("quality_type")
    if q not in L.TYPE_MAP:
        raise ValueError("quality_type must be L, M or H")
    vals = {}
    for key, (lo, hi) in LIMITS.items():
        v = p.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{key} must be a number")
        if not lo <= v <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        vals[key] = float(v)
    return q, vals


def predict(payload):
    q, v = validate(payload)
    args = (q, v["air_temperature"], v["process_temperature"], v["rotational_speed"],
            v["torque"], v["tool_wear"])
    x = L.build_row(*args)
    risk = L.failure_probability(x)
    alert = risk >= L.THRESHOLD
    share = L.type_share(L.type_scores(x))
    top = max(share, key=share.get)
    return {
        "status": "alert" if alert else "healthy",
        "risk_percent": round(risk * 100, 1),
        "threshold_percent": round(L.THRESHOLD * 100, 1),
        "likely_cause": L.TYPE_NAMES[top] if alert else None,
        "diagnosis_confidence_percent": round(share[top] * 100) if alert else None,
        "causes": [{"name": L.TYPE_NAMES[k], "percent": round(share[k] * 100)} for k in L.TYPES],
        "notes": L.watch_notes(*args),
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Send a JSON object")
            self._send(200, predict(payload))
        except (ValueError, json.JSONDecodeError) as e:
            self._send(422, {"detail": str(e)})
        except Exception:
            self._send(500, {"detail": "Prediction failed on the server"})

    def do_GET(self):
        self._send(405, {"detail": "Use POST with the six machine readings"})
