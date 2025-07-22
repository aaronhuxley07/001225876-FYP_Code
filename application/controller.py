import pandas as pd
from model import PredictionModel
from view import AppView

class AppController:
    def __init__(self):
        self.model = PredictionModel()
        self.view = AppView()

    def run(self):
        self.view.show_title()
        uploaded_file = self.view.upload_file()

        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            df_scaled, original_id, dropped_col, dropped_name, error = self.model.preprocess(df_raw)

            if error:
                self.view.show_error(error)
                return

            df_input = df_scaled.drop(columns=["txID"])
            predictions = self.model.predict(df_input)
            df_with_predictions = self.model.attach_predictions(df_scaled, predictions, original_id, dropped_col, dropped_name)

            illicit_predictions = self.view.display_predictions(df_with_predictions, original_id, dropped_name)
            shap_values = self.model.explain(df_input)
            self.view.display_explanations(df_input, shap_values, illicit_predictions)
