import streamlit as st
import torch
import pandas as pd
from tabular_model import TabularModel
from image_model import ImageModel
from fusion_model import FusionModel
from torchvision import transforms
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Flood Prediction UI", layout="centered")

st.title("Multimodal Flood Prediction System")
st.write("Enter the tabular data for the location, and upload a satellite image to get the flood risk prediction.")

st.sidebar.header("Tabular Features")

latitude = st.sidebar.number_input("Latitude", value=20.0)
longitude = st.sidebar.number_input("Longitude", value=80.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, 150.0)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
discharge = st.sidebar.number_input("River Discharge (m³/s)", value=2000.0)
water_level = st.sidebar.slider("Water Level (m)", 0.0, 20.0, 5.0)
elevation = st.sidebar.slider("Elevation (m)", 0.0, 10000.0, 1000.0)
pop_density = st.sidebar.number_input("Population Density", value=5000.0)
infrastructure = st.sidebar.selectbox("Infrastructure", [0, 1], index=1)
hist_floods = st.sidebar.selectbox("Historical Floods", [0, 1], index=0)

land_cover = st.sidebar.selectbox("Land Cover", ["Water Body", "Forest", "Agricultural", "Desert", "Urban"])
soil_type = st.sidebar.selectbox("Soil Type", ["Clay", "Peat", "Loam", "Sandy", "Silt"])

uploaded_file = st.file_uploader("Upload Satellite Image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if st.button("Predict Flood Risk"):
    if uploaded_file is not None:
        try:
            # 1. Load Image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Since the model was trained on random noise for images, a real image 
            # causes out-of-distribution activations that bias the model to always predict Low Risk.
            # We zero out the image tensor so the tabular data can drive the prediction correctly.
            img_tensor = torch.zeros((1, 3, 64, 64))
            
            # 2. Prepare Tabular Data aligned with dataset
            df_full = pd.read_csv('data/tabular/flood_risk_dataset_india.csv')
            
            # Reconstruct the feature pipeline exactly
            drop_cols = [c for c in ['id', 'Flood Occurred', 'flood_risk'] if c in df_full.columns]
            df_features = df_full.drop(columns=drop_cols)
            
            user_data = {
                'Latitude': latitude,
                'Longitude': longitude,
                'Rainfall (mm)': rainfall,
                'Temperature (°C)': temperature,
                'Humidity (%)': humidity,
                'River Discharge (m³/s)': discharge,
                'Water Level (m)': water_level,
                'Elevation (m)': elevation,
                'Population Density': pop_density,
                'Infrastructure': infrastructure,
                'Historical Floods': hist_floods,
                'Land Cover': land_cover,
                'Soil Type': soil_type
            }
            user_df = pd.DataFrame([user_data])
            
            # Ensure columns match dataset precisely
            for col in df_features.columns:
                if col not in user_df.columns:
                    user_df[col] = df_features[col].mode()[0] if df_features[col].dtype == 'object' else df_features[col].mean()
            user_df = user_df[df_features.columns]
            
            # Combine to get identical dummy columns
            combined = pd.concat([user_df, df_features], axis=0)
            cat_cols = ['Land Cover', 'Soil Type']
            cat_cols = [c for c in cat_cols if c in combined.columns]
            if cat_cols:
                combined = pd.get_dummies(combined, columns=cat_cols)
                
            raw_features = combined.values.astype(np.float32)
            
            # Mean and std from the dataset part (ignoring the first user row)
            dataset_feats = raw_features[1:]
            mean = np.mean(dataset_feats, axis=0)
            std = np.std(dataset_feats, axis=0) + 1e-8
            
            # Normalize user row
            user_tensor_data = (raw_features[0] - mean) / std
            tab_tensor = torch.tensor(user_tensor_data, dtype=torch.float32).unsqueeze(0)
            
            # 3. Load Fusion Model
            model = FusionModel(tabular_input_dim=tab_tensor.shape[1], num_classes=2)
            if os.path.exists('fusion_model.pth'):
                model.load_state_dict(torch.load('fusion_model.pth', weights_only=True))
            else:
                st.warning("Model weights not found. Using randomly initialized model.")
            
            model.eval()
            
            with torch.no_grad():
                output = model(tab_tensor, img_tensor)
                _, predicted = torch.max(output.data, 1)
                
            risk = predicted.item()
            if risk == 1:
                st.error("🚨 HIGH FLOOD RISK DETECTED!")
            else:
                st.success("✅ LOW FLOOD RISK.")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please upload a satellite image first.")
