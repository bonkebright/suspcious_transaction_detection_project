import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans # Import KMeans for potentially deployed model
from sklearn.ensemble import IsolationForest # Import IsolationForest
from sklearn.cluster import DBSCAN # Import DBSCAN

st.set_page_config(page_title="ML Assignment Deployment", layout="wide")
st.title("🚀 Anomaly Detection Deployment")

st.header("Suspicious Transaction Detection")
st.write("Enter transaction details below or upload a CSV file for anomaly detection.")

# Function to load model and preprocessor based on selection
@st.cache_resource
def load_anomaly_bundle(selected_model_name):
    model_filename = ''
    if selected_model_name == 'DBSCAN':
        model_filename = 'dbscan_model.pkl'
    elif selected_model_name == 'K-Means':
        model_filename = 'kmeans_model.pkl'

    if not model_filename:
        st.error("No model selected or model filename not defined.")
        return None, None, None, None, None # Added threshold as last None

    try:
        with open(model_filename, 'rb') as f:
            bundle = pickle.load(f)
        model = bundle['model']
        preprocessor = bundle['preprocessor']
        numerical_features = bundle['numerical_features']
        categorical_features = bundle['categorical_features']
        threshold = bundle.get('threshold') # K-Means specific
        return model, preprocessor, numerical_features, categorical_features, threshold
    except FileNotFoundError:
        st.error(f"Anomaly detection model bundle '{model_filename}' not found. Please train and save the model first.")
        return None, None, None, None, None

# Model selection dropdown
selected_model_option = st.selectbox(
    "Choose an Anomaly Detection Model:",
    ('DBSCAN', 'K-Means'),
    index=0 # Default to DBSCAN
)

anomaly_model, preprocessor, numerical_features_saved, categorical_features_saved, kmeans_threshold = load_anomaly_bundle(selected_model_option)

if anomaly_model and preprocessor and numerical_features_saved is not None and categorical_features_saved is not None:
    # Determine the type of the loaded model for specific prediction logic if needed
    model_type = type(anomaly_model).__name__

    # Option 1: Individual Transaction Input
    st.subheader("Single Transaction Anomaly Prediction")

    # Create input fields for numerical features
    input_values_num = {}
    cols_num = st.columns(len(numerical_features_saved))
    for i, feature in enumerate(numerical_features_saved):
        with cols_num[i]:
            default_value = 0.0
            if feature == 'LocalAmount':
                default_value = 100.0
            elif 'hour' in feature:
                default_value = 12 # Midday
            input_values_num[feature] = st.number_input(f"{feature} (Numerical)", value=default_value, format="%.2f", key=f"single_{feature}_num")

    # Create input fields for categorical features
    input_values_cat = {}
    cols_cat = st.columns(len(categorical_features_saved))
    for i, feature in enumerate(categorical_features_saved):
        with cols_cat[i]:
            if feature == 'TransactionChannels':
                common_channels = ['Online', 'Branch', 'ATM', 'Mobile', 'Unknown'] # Example common channels, include 'Unknown'
                input_values_cat[feature] = st.selectbox(f"{feature} (Categorical)", options=common_channels, key=f"single_{feature}_cat")
            else:
                input_values_cat[feature] = st.text_input(f"{feature} (Categorical)", value=f"test_id_example", key=f"single_{feature}_cat")

    if st.button("Detect Anomaly in Single Transaction"): # Changed button text
        if anomaly_model and preprocessor:
            single_transaction_data = pd.DataFrame([{**input_values_num, **input_values_cat}])

            required_columns = numerical_features_saved + categorical_features_saved
            for col in required_columns:
                if col not in single_transaction_data.columns:
                    st.error(f"Missing input for required feature: {col}")
                    st.stop()

            scaled_single = preprocessor.transform(single_transaction_data[required_columns])

            # Adjust prediction logic based on model type
            if model_type == 'KMeans':
                distances = anomaly_model.transform(scaled_single)
                min_distance = np.min(distances, axis=1)[0]
                if kmeans_threshold is not None:
                    st.write(f"Distance to nearest centroid: {min_distance:.2f} (Threshold: {kmeans_threshold:.2f})")
                    if min_distance > kmeans_threshold:
                         st.error("⚠️ Suspicious Transaction Detected! (K-Means distance based)")
                    else:
                         st.success("✅ Normal Transaction. (K-Means distance based)")
                else:
                    st.warning("K-Means threshold not loaded. Cannot reliably classify anomaly.")

            elif model_type == 'DBSCAN':
                # DBSCAN predict returns cluster labels (-1 for anomalies)
                pred_single = anomaly_model.fit_predict(scaled_single)
                if pred_single[0] == -1:
                    st.error("⚠️ Suspicious Transaction Detected! (DBSCAN)")
                else:
                    st.success("✅ Normal Transaction. (DBSCAN)")
            else:
                # For IsolationForest (predict -1 for anomalies)
                pred_single = anomaly_model.predict(scaled_single)
                if pred_single[0] == -1:
                    st.error("⚠️ Suspicious Transaction Detected! (Isolation Forest)")
                else:
                    st.success("✅ Normal Transaction. (Isolation Forest)")

    st.markdown("--- ")

    # Option 2: CSV File Upload
    st.subheader("Batch Anomaly Detection from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file for anomaly detection", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df_uploaded.head())

        if 'CreatedOn' in df_uploaded.columns and 'CreatedOn_hour' in numerical_features_saved:
            df_uploaded['CreatedOn_dt'] = pd.to_datetime(df_uploaded['CreatedOn'], errors='coerce')
            df_uploaded['CreatedOn_hour'] = df_uploaded['CreatedOn_dt'].dt.hour
        elif 'CreatedOn_hour' in numerical_features_saved and 'CreatedOn_hour' not in df_uploaded.columns:
            st.warning("Column 'CreatedOn' not found in uploaded CSV, and 'CreatedOn_hour' is a required numerical feature. Filling 'CreatedOn_hour' with 0 (default).")
            df_uploaded['CreatedOn_hour'] = 0

        df_uploaded.dropna(subset=[f for f in numerical_features_saved if f in df_uploaded.columns], inplace=True)

        for col in categorical_features_saved:
            if col not in df_uploaded.columns:
                st.warning(f"Column '{col}' not found in uploaded CSV. Adding column and filling with 'Unknown'.")
                df_uploaded[col] = 'Unknown'
            else:
                df_uploaded[col] = df_uploaded[col].fillna('Unknown')

        all_required_cols = numerical_features_saved + categorical_features_saved
        missing_from_uploaded = [col for col in all_required_cols if col not in df_uploaded.columns]
        if missing_from_uploaded:
            st.error(f"Error: Uploaded CSV and its processed form must contain all required features: {', '.join(all_required_cols)}. Missing: {', '.join(missing_from_uploaded)}")
            st.stop()

        X_uploaded = df_uploaded[all_required_cols]

        scaled_uploaded = preprocessor.transform(X_uploaded)

        # Adjust prediction logic based on model type for batch prediction
        if model_type == 'KMeans':
            distances_batch = anomaly_model.transform(scaled_uploaded)
            min_distances_batch = np.min(distances_batch, axis=1)
            if kmeans_threshold is not None:
                predictions_batch = np.where(min_distances_batch > kmeans_threshold, -1, 1)
            else:
                st.warning("K-Means threshold not loaded. Cannot reliably classify anomalies in batch. Defaulting to normal.")
                predictions_batch = np.ones(len(df_uploaded)) # Default to normal if no threshold
        elif model_type == 'DBSCAN':
            predictions_batch = anomaly_model.fit_predict(scaled_uploaded)
        else:
            predictions_batch = anomaly_model.predict(scaled_uploaded)

        df_uploaded['Anomaly_Prediction'] = np.where(predictions_batch == -1, "Suspicious", "Normal")

        st.write("Anomaly Detection Results:")
        st.dataframe(df_uploaded.style.apply(lambda x: ["background-color: #ffe6e6" if x["Anomaly_Prediction"] == "Suspicious" else "" for i in x], axis=1))

        num_suspicious = (df_uploaded["Anomaly_Prediction"] == "Suspicious").sum()
        st.info(f"Detected {num_suspicious} suspicious transactions out of {len(df_uploaded)} processed records.")
else:
    st.warning("Model could not be loaded. Please ensure the anomaly detection model is trained and saved in the notebook.")
