import argparse, json, pathlib
import joblib, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/model.joblib")
p.add_argument("--test", default="data/processed/test.csv")
p.add_argument("--out", default="reports/metrics.json")
a = p.parse_args()

df = pd.read_csv(a.test)
X_test, Y_test = df.drop(columns="serum_creatinine"), df["serum_creatinine"]
model = joblib.load(a.model)
pred = model.predict(X_test)
metrics = {
    "model": a.model,
    "mae": round(float(mean_absolute_error(Y_test, pred)), 4),
    "mse": round(float(mean_squared_error(Y_test, pred)), 4),
    "r2": round(float(r2_score(Y_test, pred)), 4),
}
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(metrics, indent=2))
print(f"[evaluate] {metrics} -> {a.out}")
