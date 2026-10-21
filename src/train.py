import yaml, joblib, mlflow, mlflow.sklearn, pandas as pd, os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/raw/data.csv")

x_train, _, y_train, _ = train_test_split(df[['x']], df['y'], test_size=0.2, random_state=params['seed'])

pf = PolynomialFeatures(degree=params['degree'])

x_train = pf.fit_transform(x_train)

model = LinearRegression()

mlflow.set_experiment("quadratic-regression")

with mlflow.start_run() as run:
    mlflow.log_param("degree", params['degree'])
    mlflow.log_param("data_size", params['data_size'])
    mlflow.log_param("seed", params['seed'])
    
    model.fit(x_train, y_train)
    mlflow.sklearn.log_model(model, name="model")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    
    with open("models/run_id.txt", "w") as f:
        f.write(run.info.run_id)