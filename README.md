# Multimodal Flood Risk Prediction

## Abstract
This project implements a multimodal machine learning model to predict flood risks by fusing diverse data sources. It integrates tabular environmental data (such as rainfall, river water levels, and geographical factors) with satellite imagery. The architecture utilizes a deep learning fusion model that combines a multi-layer perceptron (MLP) for tabular feature extraction and a Convolutional Neural Network (CNN) for image feature extraction. This robust approach aims to provide more accurate, real-time flood risk predictions compared to single-modality systems. A Streamlit web application is included to demonstrate real-time, interactive predictions.

## Project Structure
- `/src`: Contains all source code for data processing, model definitions, training scripts, and the Streamlit web application.
- `/data`: Contains the tabular datasets and satellite imagery.
- `/results`: Contains the saved trained model weights (`.pth` files), confusion matrices, and performance metric charts.
- `requirements.txt`: Lists all the necessary Python dependencies to run the project.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Lahari-dot/flood-prediction-project.git
   cd flood-prediction-project
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training scripts (if you want to retrain the models):**
   Ensure you are in the project root directory.
   ```bash
   python src/train_tabular.py
   python src/train_image.py
   python src/train_fusion.py
   ```

5. **Run the Streamlit Web Application:**
   Start the interactive web interface to test predictions.
   ```bash
   streamlit run src/app.py
   ```
