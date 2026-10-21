import numpy as np
import pandas as pd
import os
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

os.makedirs("data/raw", exist_ok=True)

np.random.seed(params['seed'])

xs = np.linspace(-10, 10, params['data_size'])

y = np.zeros_like(xs)
for i in range(params['degree'] + 1):
    y += i * (xs ** i)

y += np.random.normal(0, 100, params['data_size'])

df = pd.DataFrame({"x": xs, "y": y})

df.to_csv("data/raw/data.csv", index=False)