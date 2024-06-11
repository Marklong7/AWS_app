import streamlit as st
import logging
import logging.config
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up logging configuration
log_dir = Path("log_data")
log_dir.mkdir(exist_ok=True)  # Create log directory if it doesn't exist

logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(log_dir / "app.log")
                    ],
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Define the artifacts directory
artifacts_dir_name = "artifacts"
artifacts_dir = Path() / artifacts_dir_name

# Function to load models from artifacts directory with caching
@st.cache_resource
def load_models():
    models = {}
    for model_file in artifacts_dir.glob("*.pkl"):
        model_name = model_file.stem
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    return models

# Function to display the model selection interface
def model_selection(models):
    st.sidebar.header("Model Selection")
    model_names = list(models.keys())
    selected_model = st.sidebar.selectbox("Choose Model", model_names, help="Select a model to use for prediction.")
    return selected_model

# Function to load the selected model
def load_selected_model(selected_model, models):
    model = models.get(selected_model)
    if model is None:
        logger.error(f"Selected model not found: {selected_model}")
        st.error("Selected model not found!")
    else:
        logger.info(f"Loaded selected model: {selected_model}")
    return model

# Function to display the prediction interface
def prediction_interface(model):
    st.header("Make Prediction")
    st.markdown("Select a sample to make a prediction based on its features.")

    # Define testing examples
    testing_samples = {
        "Negative Sample": [1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
                            3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
                            8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
                            3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
                            1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01],
        "Positive Sample": [1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
                            4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
                            1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
                            1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
                            1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]
    }

    # User selects which sample to use
    selected_sample = st.radio("Select Sample:", list(testing_samples.keys()), help="Choose a sample to see its prediction.")

    # Get the features of the selected sample
    sample_features = np.array(testing_samples[selected_sample])

    # Calculate the mean of each feature across both samples
    mean_features = np.mean(list(testing_samples.values()), axis=0)

    # Calculate the percentage difference of each feature's value from its respective mean
    percent_difference = ((sample_features - mean_features) / mean_features) * 100

    # Plot the histogram showing the percentage difference
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(percent_difference)), percent_difference, color='skyblue', edgecolor='black')
    plt.title(f"Percentage Difference of {selected_sample} Features from Mean")
    plt.xlabel("Feature Index")
    plt.ylabel("Percentage Difference (%)")
    plt.xticks(range(len(percent_difference)))
    st.pyplot(plt)

    # Make prediction based on the original features (not standardized)
    try:
        prediction = model.predict(sample_features.reshape(1, -1))
        logger.info(f"Made prediction using {selected_sample}")

        # Display the prediction result
        if prediction[0] == 0:
            predicted_label = "Malignant"
        else:
            predicted_label = "Benign"

        actual_label = "Malignant" if selected_sample == "Negative Sample" else "Benign"

        st.subheader("Prediction Result:")
        st.write("Predicted Label:", predicted_label)
        st.write("Actual Label:", actual_label)

        if predicted_label == actual_label:
            st.success("Prediction is CORRECT!")
        else:
            st.error("Prediction is WRONG!")

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        st.error("Error making prediction. Please try again.")

# Main function to run the app
def main():
    st.title("Breast Cancer Prediction App")
    st.markdown("""
        This app uses machine learning models to predict breast cancer based on sample data.
        Select a model from the sidebar and then choose a sample to see the prediction.
    """)

    # Load models
    try:
        models = load_models()
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error("Error loading models. Please check the artifacts directory.")
        return
    
    if not models:
        logger.warning("No models found in the artifacts directory.")
        st.warning("No models found in the artifacts directory.")
        return
    
    # Display model selection interface
    selected_model = model_selection(models)
    
    # Load selected model
    model = load_selected_model(selected_model, models)
    
    if model is not None:
        # Display prediction interface
        prediction_interface(model)

if __name__ == "__main__":
    main()
