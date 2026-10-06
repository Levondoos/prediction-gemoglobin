import argparse, pathlib, time
import joblib, pandas as pd
import xgboost as xgb

p = argparse.ArgumentParser()
p.add_argument("--train", default="data/processed/train.csv")
p.add_argument("--model-out", default="models/model.joblib")
p.add_argument("--n-estimators", type=int, default=100)
p.add_argument("--learning-rate", type=float, default=0.01)
p.add_argument("--max-depth", type=int, default=3)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
df = pd.read_csv(a.train)
X_train, Y_train = df.drop(columns="serum_creatinine"), df["serum_creatinine"]

t0 = time.time()

xgbb = xgb.XGBRegressor( n_estimators=a.n_estimators, learning_rate = a.learning_rate,  max_depth=a.max_depth, random_state=a.seed)
xgbb.fit(X_train, Y_train)

pathlib.Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(xgbb, a.model_out)
print(f"[train] n_estimators={a.n_estimators} max_depth={a.max_depth} learning_rate={a.learning_rate} "
      f"time={time.time()-t0:.1f}s -> {a.model_out}")