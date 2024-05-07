import numpy as np
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import standard scaler from scikit-learn

model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load your dataset
df = pd.read_csv('uneguicar1.csv')

def car_price_prediction(input_data):
    input_changed = np.array(input_data).reshape(1, -1)
    std_input = scaler.transform(input_changed)
    prediction = model.predict(std_input)
    return "Estimated car price: " + str(prediction[0])

def plot_feature_importance():
    feature_importance = model.feature_importances_
    feature_names = ['Model', 'Manufactured Year', 'Entry Year', 'Engine Size', 'Transmission', 'Fuel Type', 'Mileage']
    
    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_importance)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for Car Price Prediction')
    return fig

def plot_time_series(df, manufacturers):
    fig, ax = plt.subplots()
    for manufacturer in manufacturers:
        filtered_df = df[df['manufacturers'] == manufacturer]
        average_prices = filtered_df.groupby('manufactured_years')['prices'].mean()
        ax.plot(average_prices.index, average_prices.values, label=manufacturer)
    ax.set_xlabel('Manufactured Year')
    ax.set_ylabel('Average Price')
    ax.set_title('Price vs. Manufactured Year Time Series')
    ax.legend()
    return fig

def main():
    st.title("Car: Prius Price Prediction App")
    
    model_input = st.text_input('**Model:** {Prius 10: **1**, Prius 20: **2**, Prius 30: **3**, Prius 40: **4**, Other: **0**}')
    manufactured_year = st.text_input('Manufactured Year')
    entry_year = st.text_input('Entry Year')
    engine_size = st.text_input('Engine Size')
    transmission = st.text_input('**Transmission:** {Automat: **0**}')
    fuel_types = st.text_input('**Fuel Type:** {Hybrid: **0**, Petrol: **1**, Gas: **2**, Diesel: **3**}')
    mileage = st.text_input('Mileage')
    
    pred_price = ''
    
    if st.button('Check Estimated Price'):
        pred_price = car_price_prediction([model_input, manufactured_year, entry_year, engine_size, transmission, fuel_types, mileage])
        
    st.success(pred_price)
    
    st.subheader("Feature Importance")
    fig = plot_feature_importance()
    st.pyplot(fig)
    
    st.sidebar.markdown("<span style='color:green'>Time Series: Price vs. Manufactured Year</span>", unsafe_allow_html=True)
    manufacturers = st.sidebar.multiselect('Select Manufacturers', df['manufacturers'].unique())
    
    if manufacturers:
        time_series_fig = plot_time_series(df, manufacturers)
        st.sidebar.pyplot(time_series_fig)
    
    st.sidebar.markdown("<span style='color:green'>Possible Cars in Price Range /in millions/</span>", unsafe_allow_html=True)
    min_price = st.sidebar.number_input("Minimum Price", value=0)
    max_price = st.sidebar.number_input("Maximum Price", value=100000)
    possible_cars = df[(df['prices'] >= min_price) & (df['prices'] <= max_price)]['manufacturers'].unique()
    st.sidebar.write("Possible cars within the price range:")
    for car in possible_cars:
        st.sidebar.markdown(f'<span style="color:{"green"}">{car}</span>', unsafe_allow_html=True)
    
    st.subheader("Car Listings Website")
    st.write("Check out the latest car listings at [Unegui.mn](https://www.unegui.mn/avto-mashin/-avtomashin-zarna/)")
    
    #add comment
    st.subheader("Comments")
    user_comment = st.text_area("Leave your comment here 😊:")
    if st.button("Submit Comment"):
        # You can add code here to handle the submitted comment, such as saving it to a database or displaying it
        st.success("Comment submitted successfully! 😊")

        
if __name__ == '__main__':
    main()
