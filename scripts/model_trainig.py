import os
import joblib
import logging
import json
import dvc.api
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        X_train=X_train.drop(['TransactionMonth'], axis=1)
        X_test=X_test.drop(['TransactionMonth'], axis=1)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Ensure necessary directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)

    def evaluate(self, model_name, predictions):
        """Evaluates model performance using regression metrics"""
        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = mean_squared_error(self.y_test, predictions, squared=False)
        r2 = r2_score(self.y_test, predictions)

        logging.info(f"📊 Model Performance: {model_name}")
        logging.info(f"➡ MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        
        return {"model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    def train_model(self, model, model_name):
        """Trains and evaluates a given model."""
        logging.info(f"Training {model_name}...")
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        # Evaluate Model
        metrics = self.evaluate(model_name, predictions)

        # Save Model
        model_path = f"models/{model_name}.pkl"
        joblib.dump(model, model_path)
        logging.info(f"Model {model_name} saved at {model_path}")

        # Save Metrics for DVC Tracking
        metrics_path = f"metrics/{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        logging.info(f"Metrics saved at {metrics_path}")

    def train_all(self):
        """Trains all models"""
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        }

        for name, model in models.items():
            self.train_model(model, name)

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor

    # Load data
    preprocessor = DataPreprocessor("data/insurance_data.csv", target_column="TotalClaims")
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    # Train models
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_all()

    # DVC Experiment Tracking (Optional - Run Manually if Needed)
    logging.info("Adding models and metrics to DVC...")
    os.system("dvc add models/")
    os.system("dvc add metrics/")
    os.system("git add models.dvc metrics.dvc")
    os.system('git commit -m "Added trained models and evaluation metrics"')
    os.system("dvc push")
