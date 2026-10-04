import pandas as pd
import argparse, pathlib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/raw/dataset.csv")
p.add_argument("--outdir", default="data/processed")
a = p.parse_args()

df = pd.read_csv(a.input)

df = df[['age', 'al', 'bu', 'sg', 'appet', 'ane', 'sc', 'classification']]
df.columns = ['age', 'albumin', 'blood_urea', 'specific_gravity', 'appetite', 'aanemia', 'serum_creatinine','class']

df['class'] = df['class'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})
df['class'] = df['class'].map({'ckd': 1, 'not ckd': 0})
df['class'] = pd.to_numeric(df['class'], errors='coerce')

cat_cols = [col for col in df.columns if (df[col].dtype == 'str')]
num_cols = [col for col in df.columns if (df[col].dtype != 'str')]

df_v = df[df['class'] == 1].copy()

df_vt = df_v[(df_v['blood_urea'] >= 0) & (df_v['blood_urea'] <= 185)]
df_v = pd.concat([df_vt, df_v[df_v['blood_urea'].isna()]], axis=0)

df_vt = df_v[(df_v['serum_creatinine'] >= 0) & (df_v['serum_creatinine'] <= 7)]
df_v = pd.concat([df_vt, df_v[df_v['serum_creatinine'].isna()]], axis=0)

df_f = pd.concat([df_v, df[df['class'] == 0]], axis=0)

df_test = df_f.dropna().sample(frac=0.30, random_state = 42).reset_index(drop=True)
df_train = pd.concat([df_f, df_test]).drop_duplicates(keep=False)
#заполнение пропусков
df_train = df_train.dropna(subset=['serum_creatinine'])

dff_o = df_train[df_train['class'] == 1].copy()
dff_p = df_train[df_train['class'] == 0].copy()

for i in ['age', 'blood_urea']:
  dff_o[i] = dff_o[i].fillna(dff_o[i].mean())
  dff_p[i] = dff_p[i].fillna(dff_p[i].mean())

dff_o['specific_gravity'] = dff_o['specific_gravity'].fillna(1.025)
dff_p['specific_gravity'] = dff_p['specific_gravity'].fillna(1.01)

dff_o['albumin'] = dff_o['albumin'].fillna(0)
dff_p['albumin'] = dff_p['albumin'].fillna(2)

for i in cat_cols:
  dff_p[i] = dff_p[i].fillna(dff_p[i].mode()[0])
  dff_o[i] = dff_o[i].fillna(dff_o[i].mode()[0])
f_d = pd.concat([dff_o, dff_p], ignore_index=True)
#нормализация
df_ob = pd.concat([df_test, f_d], axis=0)
df_ob = df_ob.drop('class', axis=1)
le = LabelEncoder()
for i in cat_cols:
  df_ob[i] = le.fit_transform(df_ob[i])

mms = MinMaxScaler()
ss = StandardScaler() 
for i in ['age', 'specific_gravity', 'albumin']:
  df_ob[i] = ss.fit_transform(df_ob[[i]])
for i in [ 'blood_urea']:
  df_ob[i] = mms.fit_transform(df_ob[[i]])

net = df_test.shape[0]
test = df_ob[:net]
train = df_ob[net:]

outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
train.to_csv(outdir / "train.csv", index=False)
test.to_csv(outdir / "test.csv", index=False)
print(train.head())