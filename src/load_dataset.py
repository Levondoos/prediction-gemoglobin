import argparse, pathlib
import kagglehub
from kagglehub import KaggleDatasetAdapter

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/raw/dataset.csv")
a = p.parse_args()

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "mansoordaku/ckdisease",
  "kidney_disease.csv")

pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)