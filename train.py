import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load the data
df = pd.read_csv(r"C:\Users\Aman Sinha\Desktop\AI ML 30 days\Day26\data\student_scores.csv")
X= df[["Hours"]]
y= df[["Scores"]]

# Set Mlflow tracking URI (Local or remote)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("StudentScore-Predictor")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X,y)

    y_pred = model.predict(X)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mean_squared_error(y,y_pred))
    mlflow.log_metric("r2_score", r2_score(y,y_pred))

    #save the model with the metadata
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=[[5]],
        registered_model_name="StudentScoreModel"
    )
