import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# connect to the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Optional: create experiment
mlflow.set_experiment("LinearRegression-Experiment")

# Sample data
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mean_squared_error(y, y_pred))
    mlflow.log_metric("r2_score", r2_score(y, y_pred))

    # Better logging using latest syntax
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=[[5]],  # Helps generate schema
        registered_model_name=None  # Skip model registry for now
    )
