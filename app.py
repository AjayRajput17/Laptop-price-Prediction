import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model and the dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set up the title for the Streamlit app
st.title("Laptop Price Predictor")

# Section for input fields

# Laptop brand selection
company = st.selectbox('Brand', df['Company'].unique())

# Laptop type selection
type = st.selectbox('Type', df['TypeName'].unique())

# RAM selection
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight of the laptop (in kg)
weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.0)

# Touchscreen selection
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS selection
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size selection
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)

# Screen resolution selection
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
     '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)

# CPU brand selection
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD selection (in GB)
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD selection (in GB)
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU brand selection
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS selection
os = st.selectbox('OS', df['os'].unique())

# Button to trigger price prediction
if st.button('Predict Price'):

    # Convert categorical values to numerical values for model prediction
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Extract screen resolution and calculate PPI (Pixels Per Inch)
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = np.sqrt(X_res**2 + Y_res**2) / screen_size

    # Prepare the input features as a numpy array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, 12)

    # Predict the price using the model pipeline
    predicted_price = np.exp(pipe.predict(query)[0])

    # Display the predicted price
    st.title(f"The predicted price of this configuration is ${predicted_price:,.2f}")
