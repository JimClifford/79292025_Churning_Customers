import streamlit as st
import pandas as pd
import pickle
# import tensorflow
# import scikeras

model_path = 'best_model.pkl'
scaler_path = 'scaler.pkl'

loaded_model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
accuracy_of_model = 82.55
# Function to preprocess user input
def preprocess_input(user_data, scaler):
    # Assuming features are stored in a DataFrame named input_data
    input_data = pd.DataFrame(user_data, index=[0])

    # Use the loaded scaler to transform the input data
    scaled_input_data = scaler.transform(input_data)

    return scaled_input_data

# Function to predict player rating
def predict_churn(scaled_input_data, model):
    # Assuming the model takes the scaled input data
    user_churn = model.predict(scaled_input_data)
    return user_churn

# Create a Streamlit web app
st.title("Customer Churn Predictor")

# Create input fields for player data
st.write("Enter Player Data:")
user_data = {}  # Create a dictionary to store user input

# Define the feature names in Xtest
features = ['TotalCharges','MonthlyCharges','tenure','Contract','PaymentMethod','TechSupport','OnlineSecurity','gender','InternetService','OnlineBackup']

# Loop through the features and create input fields
for feature in features:
    if (feature == 'gender'):
        user_data[feature] = st.number_input(f"{feature.capitalize()}: Enter 0(Female) or 1(Male)", min_value=0, max_value=1)
     
    
    elif (feature == "InternetService"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}: Enter 0(No) or 1(Fiber optic) or 2(Yes)", min_value=0, max_value=2)
    
    elif (feature == "OnlineSecurity"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}: Enter 0(No) or 1(No internet service) or 2(Yes)", min_value=0, max_value=2)
    

    elif (feature == "OnlineBackup"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}: Enter 0(No) or 1(No internet service) or 2(Yes)", min_value=0, max_value=2)
    
    
    elif (feature == "TechSupport"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}: Enter 0(No) or 1(No internet service) or 2(Yes)", min_value=0, max_value=2)
    
    
    elif (feature == "Contract"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}: Enter 0(Month-to-month) or 1(One year) or 2(Two year)", min_value=0, max_value=2)
    
    
    elif (feature == "PaymentMethod"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}: Enter 0(Bank transfer (automatic)) or 1(Credit card (automatic)) or 2(Electronic check) or 3(Mailed check)", min_value=0, max_value=3)
    
    
    elif (feature == "TotalCharges"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}", min_value=0.00, max_value=10000.00)
    
    elif (feature == "MonthlyCharges"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}", min_value=0.00, max_value=200.00)
    
    elif (feature == "tenure"):
        user_data[feature] = st.number_input(f"{feature.capitalize()}", min_value=0.0, max_value=72.0)
    

    


input_data = preprocess_input(user_data,scaler)
if st.button("Predict"):
    # Convert the input data to a DataFrame for prediction
    #input_data = pd.DataFrame(player_data, index=[0])

    # Call the prediction function
    user_churn = predict_churn(input_data,loaded_model)
    user_churn = user_churn.astype(int)
    # from the analysis of dataset, please look through model training file
    if user_churn[0] == 1:
        prediction = "Yes"
    else:
        prediction = "No"
    
    # Display the predicted player rating
    st.write(f"Will User Churn: {prediction}")
    st.write(f"Model Accuracy: {accuracy_of_model}")
