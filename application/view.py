import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

class AppView:
    def show_title(self):
        st.title("Cryptocurrency Fraud Detector")

    def upload_file(self):
        return st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])

    def show_error(self, msg):
        st.error(msg)

    def display_predictions(self, df_with_predictions, original_id_col, dropped_col_name):
        df_with_predictions = df_with_predictions[[original_id_col, dropped_col_name] +
                                       [col for col in df_with_predictions.columns if col not in [original_id_col, dropped_col_name]]]

        st.subheader(f"All Predictions ({len(df_with_predictions)} Transactions)")

        rows_per_page = 100
        total_pages = (len(df_with_predictions) - 1) // rows_per_page + 1
        if "all_page" not in st.session_state:
            st.session_state["all_page"] = 1

        page = st.session_state["all_page"]
        start = (page - 1) * rows_per_page
        end = start + rows_per_page
        page_df = df_with_predictions.iloc[start:end]

        def color_row(row):
            color = '#FF0000' if row['Prediction'] == 'Illicit' else '#008000'
            return [f'background-color: {color}'] * len(row)

        st.dataframe(page_df.style.apply(color_row, axis=1), use_container_width=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Prev", key="all_prev") and page > 1:
                st.session_state["all_page"] -= 1
        with col2:
            st.markdown(f"<div style='text-align: center; padding-top: 8px;'>Page {page} of {total_pages}</div>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️", key="all_next") and page < total_pages:
                st.session_state["all_page"] += 1

        csv_all = df_with_predictions.to_csv(index=False).encode("utf-8")
        st.download_button("📁 Download All Predictions CSV", csv_all, "all_predictions.csv", "text/csv")

        illicit_predictions = df_with_predictions[df_with_predictions["Prediction"] == "Illicit"]
        st.subheader(f"Illicit Predictions Only ({len(illicit_predictions)} Transactions)")
        st.dataframe(illicit_predictions)

        csv_illicit = illicit_predictions.to_csv(index=False).encode("utf-8")
        st.download_button("📁 Download Illicit Predictions CSV", csv_illicit, "illicit_predictions.csv", "text/csv")

        return illicit_predictions

    def display_explanations(self, df_input, shap_values, illicit_predictions):
        if illicit_predictions.empty:
            st.info("No illicit transactions were detected.")
            return

        shap_vals_array = shap_values.values
        explanations = []
        transaction_ids = []

        for idx in illicit_predictions.index:
            row_shap = shap_vals_array[idx]
            abs_shap = np.abs(row_shap)
            top3_idx = np.argsort(abs_shap)[-3:][::-1]
            top3_features = df_input.columns[top3_idx]
            top3_shap_values = row_shap[top3_idx]

            expl_parts = [f"high {feat}" if val > 0 else f"low {feat}"
                          for feat, val in zip(top3_features, top3_shap_values)]

            explanation = f"Transaction #{idx} was flagged primarily because of " + \
                          ", ".join(expl_parts[:-1]) + f", and {expl_parts[-1]}."

            explanations.append({
                "Transaction ID": idx,
                "Feature 1": top3_features[0],
                "SHAP 1": top3_shap_values[0],
                "Feature 2": top3_features[1],
                "SHAP 2": top3_shap_values[1],
                "Feature 3": top3_features[2],
                "SHAP 3": top3_shap_values[2],
                "Explanation": explanation
            })
            transaction_ids.append(str(idx))

        explanation_df = pd.DataFrame(explanations).drop(columns=["SHAP 1", "SHAP 2", "SHAP 3"])
        st.subheader("Summary Table of Top Contributing Features and Explanations")
        st.dataframe(explanation_df)

        csv_summary = explanation_df.to_csv(index=False).encode("utf-8")
        st.download_button("📁 Download Summary CSV", csv_summary, "summary_explanations.csv", "text/csv")

        search_term = st.text_input("Search transaction ID")
        if search_term:
            filtered_ids = [tx for tx in transaction_ids if search_term in tx]
        else:
            filtered_ids = transaction_ids

        if filtered_ids:
            selected_tx = st.selectbox("Select a transaction ID to see its detailed SHAP bar plot", filtered_ids)
            self._display_shap_bar_plot(shap_vals_array, explanations, selected_tx)
        else:
            st.warning("No transactions match your search.")

        if st.checkbox("Advanced: Show SHAP Heatmap for Illicit Transactions"):
            self._show_shap_heatmap(df_input, shap_vals_array, illicit_predictions)

    def _display_shap_bar_plot(self, shap_vals_array, explanations, selected_tx):
        selected_tx_int = int(selected_tx)
        for expl in explanations:
            if expl["Transaction ID"] == selected_tx_int:
                sel_feats = [expl["Feature 1"], expl["Feature 2"], expl["Feature 3"]]
                sel_shap_vals = [expl["SHAP 1"], expl["SHAP 2"], expl["SHAP 3"]]
                break

        fig, ax = plt.subplots()
        ax.barh(sel_feats[::-1], sel_shap_vals[::-1],
                color=['red' if x < 0 else 'green' for x in sel_shap_vals[::-1]])
        ax.set_title(f"Top 3 Feature Contributions for Transaction #{selected_tx_int}")
        ax.set_xlabel("SHAP Value (Impact on prediction)")
        st.pyplot(fig)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight")
        img_buf.seek(0)
        st.download_button("🖼️ Download Contribution Plot PNG", img_buf, f"transaction_{selected_tx}_shap.png", "image/png")

    def _show_shap_heatmap(self, df_input, shap_vals_array, illicit_predictions):
        st.subheader("SHAP Heatmap (Top Features of Illicit Transactions)")
        illicit_indices = illicit_predictions.index.tolist()
        shap_illicit = shap_vals_array[illicit_indices, :]

        num_feats_to_show = st.slider("Number of top features to show", 3, 30, 10)
        mean_abs_shap = np.abs(shap_vals_array).mean(axis=0)
        top_feature_indices = np.argsort(mean_abs_shap)[-num_feats_to_show:][::-1]
        top_feature_names = df_input.columns[top_feature_indices]
        shap_heatmap_data = shap_illicit[:, top_feature_indices]

        heatmap_df = pd.DataFrame(shap_heatmap_data, columns=top_feature_names, index=illicit_indices)
        heatmap_df.index.name = "Transaction Index"
        heatmap_df = heatmap_df.T

        fig, ax = plt.subplots(figsize=(min(0.6 * len(illicit_indices), 15), 0.5 * num_feats_to_show + 2))
        sns.heatmap(heatmap_df, cmap='RdBu_r', center=0, cbar_kws={"label": "SHAP Value"}, ax=ax)
        ax.set_title("SHAP Heatmap (Top Features of Illicit Transactions)")
        ax.set_xlabel("Transaction Index")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

        heat_buf = io.BytesIO()
        fig.savefig(heat_buf, format="png", bbox_inches="tight")
        heat_buf.seek(0)
        st.download_button("🖼️ Download SHAP Heatmap PNG", heat_buf, "shap_heatmap.png", "image/png")