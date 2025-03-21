from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, y_test, predictions):
        self.y_test = y_test
        self.predictions = predictions

    def evaluate(self, model_name):
        """Evaluates model performance using regression metrics"""
        mae = mean_absolute_error(self.y_test, self.predictions)
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = mean_squared_error(self.y_test, self.predictions, squared=False)
        r2 = r2_score(self.y_test, self.predictions)

        print(f"📊 Model Performance: {model_name}")
        print(f"➡ MAE: {mae:.4f}")
        print(f"➡ MSE: {mse:.4f}")
        print(f"➡ RMSE: {rmse:.4f}")
        print(f"➡ R² Score: {r2:.4f}")
        print("=" * 40)

        return {"model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
