"""One-off script: turns pm_xgb.json + pm_meta.joblib into lib/model_lite.json.
The lite file needs no xgboost, scikit-learn, numpy or pandas to run."""
import json, math, re, sys, warnings
import joblib
warnings.simplefilter("ignore")

xgb_path, meta_path, out_path = sys.argv[1:4]

# ---- XGBoost: keep only what prediction needs ----
m = json.load(open(xgb_path))["learner"]
bs = float(re.findall(r"[-+0-9.eE]+", m["learner_model_param"]["base_score"])[0])
base_margin = math.log(bs / (1 - bs))
trees = []
for t in m["gradient_booster"]["model"]["trees"]:
    trees.append({"l": t["left_children"], "r": t["right_children"],
                  "f": t["split_indices"], "v": t["split_conditions"],   # leaves store their value here
                  "d": t["default_left"]})

# ---- Random forest (one per failure type): store leaf probability of class 1 ----
meta = joblib.load(meta_path)
rf = {}
for name, est in zip(meta["types"], meta["type_model"].estimators_):
    assert list(est.classes_) == [0, 1], est.classes_
    forest = []
    for tr in est.estimators_:
        tt = tr.tree_
        val = tt.value[:, 0, :]
        p1 = (val[:, 1] / val.sum(axis=1)).tolist()
        forest.append({"l": tt.children_left.tolist(), "r": tt.children_right.tolist(),
                       "f": tt.feature.tolist(), "t": tt.threshold.tolist(), "p": p1})
    rf[name] = forest

json.dump({"features": meta["features"], "types": meta["types"],
           "threshold": float(meta["threshold"]), "base_margin": base_margin,
           "xgb": trees, "rf": rf}, open(out_path, "w"), separators=(",", ":"))
print("wrote", out_path)
