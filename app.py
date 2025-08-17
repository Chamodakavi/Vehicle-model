import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the pre-trained model and encoder
model_file = 'decision_tree_model.pkl'

# Load the saved model
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Load the encoder used for categorical data transformation
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Placeholder for one-hot encoder fitting
def fit_encoder(data):
    encoder.fit(data)
    return encoder

# Streamlit title
st.title("Vehicle Price Prediction")

# Input fields for vehicle features
st.subheader("Enter the vehicle details")

year = st.number_input("Year of Manufacture", min_value=1900, max_value=2025, value=2020)
mileage = st.number_input("Mileage (in miles)", min_value=0, value=20.0)
model_input = st.text_input("Model", "Wagoneer")
fuel_input = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])

# Preparing the input data for prediction
input_data = pd.DataFrame({
    'year': [year],
    'mileage': [mileage],
    'model': [model_input],
    'fuel': [fuel_input]
})

# Encode the categorical features using the previously fitted encoder
encoded_input = encoder.transform(input_data[['model', 'fuel']])
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['model', 'fuel']))

# Combine numeric columns with encoded categorical columns
final_input = pd.concat([input_data[['year', 'mileage']], encoded_df], axis=1)

# Align the input columns with the trained model's expected input
final_input = final_input.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict the price using the trained model
if st.button("Predict Price"):
    predicted_price = model.predict(final_input)
    st.write(f"The predicted price for the vehicle is: ${predicted_price[0]:,.2f}")
