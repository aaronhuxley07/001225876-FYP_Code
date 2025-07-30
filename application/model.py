import xgboost as xgb
import shap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load

NUM_FEATURES = 165

class PredictionModel:
    def __init__(self):
        self.model = self._load_model()
        self.explainer = self._load_explainer()

    def _load_model(self):
        return load("models/model.pkl")

    def _load_explainer(self):
        shap_model = load("models/explainer.pkl")
        return shap.TreeExplainer(shap_model)
        
    def preprocess(self, df):
        original_columns = df.columns.tolist()
        if len(original_columns) < 2:
            return None, None, None, None, "CSV must have at least two columns."

        original_id = original_columns[0]
        dropped_name = original_columns[1]

        df = df.rename(columns={original_id: "txID"})
        dropped_col = df[dropped_name]
        df = df.drop(columns=[dropped_name])

        features = df.drop(columns=["txID"])
        if features.shape[1] != NUM_FEATURES:
            return None, None, None, None, f"Expected {NUM_FEATURES} features, got {features.shape[1]}"

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)
        df_scaled = pd.DataFrame(scaled, columns=features.columns)
        df_scaled.insert(0, "txID", df["txID"])

        return df_scaled, original_id, dropped_col, dropped_name, None

    def predict(self, df_input):
        # Booster requires DMatrix for prediction
        dmatrix = xgb.DMatrix(df_input.drop(columns=["txID"]))
        return self.model.predict(dmatrix)

    def explain(self, df_input):
        return self.explainer(df_input.drop(columns=["txID"]))

    def attach_predictions(self, df_scaled, predictions, original_id, dropped_col, dropped_name):
        df_with_predictions = df_scaled.copy()
        df_with_predictions["Prediction"] = ["Illicit" if p == 1 else "Licit" for p in predictions]
        df_with_predictions = df_with_predictions.rename(columns={"txID": original_id})
        df_with_predictions[dropped_name] = dropped_col
        return df_with_predictions
