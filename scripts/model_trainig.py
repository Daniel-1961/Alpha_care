import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from evaluate import ModelEvaluator
import dvc.api

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, model, model_name):
        """Trains and evaluates a given model"""
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        evaluator = ModelEvaluator(self.y_test, predictions)
        evaluator.evaluate(model_name)

        joblib.dump(model, f"models/{model_name}.pkl")

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

    preprocessor = DataPreprocessor("data/insurance_data.csv", target_column="TotalClaims")
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_all()
