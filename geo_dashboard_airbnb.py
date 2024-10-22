import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import joblib 
from sklearn.preprocessing import LabelEncoder  

# ------------------ Paths to Files and Model ------------------ #
best_model_path = r'D:\GUVI_Projects\My_Projects\best_model.joblib'
cleaned_data_path = r'D:\GUVI_Projects\My_Projects\cleaned_airbnb_data.csv'
file_path = r'D:\GUVI_Projects\My_Projects\updated_cleaned_airbnb_data.csv'

# ------------------ Load Dataset ------------------ #
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    if 'last_scraped' in data.columns:
        data['last_scraped'] = pd.to_datetime(data['last_scraped'], dayfirst=True)
    if 'price' in data.columns:
        data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float).fillna(0)
    if 'latitude' in data.columns:
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce').fillna(0)
    if 'longitude' in data.columns:
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce').fillna(0)
    return data

# ------------------ Load Trained Model ------------------ #
@st.cache_data
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# ------------------ Encode Categorical Features ------------------ #
def encode_features(df, feature):
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])
    return df, encoder

# ------------------ Geospatial Visualization with Pydeck ------------------ #
def geospatial_visualization(df):
    st.title("Geospatial Visualization of Airbnb Listings")

    # Pydeck for map visualization
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=df['latitude'].mean(),
            longitude=df['longitude'].mean(),
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=df[['latitude', 'longitude']],
                get_position='[longitude, latitude]',
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
        ],
    ))

    st.header("Heat Map of Listings")
    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        data=df[['latitude', 'longitude']],
        get_position='[longitude, latitude]',
        radius=200,
        threshold=0.3
    )
    
    st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=11,
        pitch=50,
    )))

# ------------------ Price Analysis with Colorful Charts ------------------ #
def price_analysis(df):
    st.title("Price Analysis")
    st.header("Price Trends Over Time")

    # Colorful Line Chart using Plotly
    fig = px.line(df, x='last_scraped', y='price', title="Price Trends", color_discrete_sequence=px.colors.sequential.Plasma)
    st.plotly_chart(fig)

    st.header("Price Distribution by Property Type")
    property_type = st.selectbox("Select Property Type", df['property_type'].unique())
    filtered_data = df[df['property_type'] == property_type]

    # Boxplot with Seaborn (for more colorful chart)
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_data, y='price', palette='Set2', ax=ax)
    ax.set_title(f"Price Distribution for {property_type}")
    st.pyplot(fig)

# ------------------ Availability Analysis ------------------ #
def availability_analysis(df):
    st.title("Availability Analysis")
    st.header("Seasonal Availability")

    df['season'] = pd.cut(df['availability_365'], bins=[-1, 90, 180, 270, 365],
                          labels=['Low', 'Moderate', 'High', 'Very High'])
    season_counts = df['season'].value_counts().sort_index()

    # Colorful Bar Chart using Plotly
    fig = px.bar(season_counts, x=season_counts.index, y=season_counts.values,
                 title="Availability by Season", color=season_counts.index, color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig)

# ------------------ Advanced Visualizations ------------------ #
def advanced_visualizations(df):
    st.title("Advanced Visualizations of Airbnb Listings")
    
    # Pair Plot
    st.header("Pair Plot of Price and Other Variables")
    pair_plot_fig = sns.pairplot(df[['price', 'number_of_reviews', 'availability_365', 'latitude', 'longitude']])
    st.pyplot(pair_plot_fig)
    st.write("This pair plot visualizes relationships between price and other key variables. Look for trends or clusters that might indicate pricing strategies based on reviews or location.")

    # Sunburst Chart
    st.header("Sunburst Chart of Property Types and Prices")
    sunburst_fig = px.sunburst(df, path=['neighbourhood_cleansed', 'property_type'], values='price',
                                title="Sunburst Chart of Property Types and Prices")
    st.plotly_chart(sunburst_fig)
    st.write("The sunburst chart shows the hierarchical relationship between neighborhoods and property types concerning their average prices. This visualization can help identify which property types dominate specific neighborhoods.")

    # Choropleth Map
    st.header("Choropleth Map of Average Prices by Neighbourhood")
    choropleth_fig = px.choropleth(df, locations='neighbourhood_cleansed', locationmode='country names',
                                    color='price', hover_name='neighbourhood_cleansed',
                                    title="Choropleth Map of Average Prices by Neighbourhood")
    st.plotly_chart(choropleth_fig)
    st.write("This choropleth map visualizes average prices by neighborhood. Darker shades indicate higher average prices, which may correlate with factors like location and amenities.")

    # Box and Violin Plot
    st.header("Price Distribution with Box and Violin Plot")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='property_type', y='price', data=df, palette="Set2")
    sns.violinplot(x='property_type', y='price', data=df, inner='quartile', color='lightgray')
    plt.title("Box and Violin Plot of Price Distribution by Property Type")
    st.pyplot(plt)
    st.write("This combined box and violin plot visualizes price distributions across property types. The violin plot provides a density estimate of the price distribution, highlighting variations within types.")

# ------------------ Predictive Analysis ------------------ #



# Load your DataFrame 
df = pd.read_csv("D:\GUVI_Projects\My_Projects\cleaned_airbnb_data.csv")

def calculate_sums(df):
    df['last_scraped_days'] = (df['last_scraped'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    
    sums = {
        'total_reviews_per_month': df['reviews_per_month'].sum(),
        'total_last_scraped_days': df['last_scraped_days'].sum(),
        'total_bedrooms': df['bedrooms'].sum(),
        'total_latitude': df['latitude'].sum(),
        'total_longitude': df['longitude'].sum(),
        'total_price_per_bedroom': df['price_per_bedroom'].sum(),
    }
    
    return sums

def predictive_analysis(df, model=None, scaler=None):
    st.title("Predictive Analysis: Airbnb Price Prediction")

    st.header("Enter Feature Values")
    
    # Define the input features
    user_inputs = {
        "Reviews Per Month": st.number_input("Reviews Per Month", min_value=0.0, step=0.1),
        "Last Scraped Date": st.date_input("Last Scraped Date"),
        "Number of Bedrooms": st.number_input("Number of Bedrooms", min_value=0, step=1),
        "Property Type": st.selectbox("Property Type", options=df['property_type'].unique()),
        "Latitude": st.number_input("Latitude"),
        "Longitude": st.number_input("Longitude"),
        "Price Per Bedroom": st.number_input("Price Per Bedroom", min_value=0.0, step=0.1),
        "Name (ID)": st.number_input("Property Name (ID)", min_value=0, step=1)  # Added name input as integer
    }

    # Convert last scraped date to days since epoch for model input
    last_scraped_days = (pd.to_datetime(user_inputs["Last Scraped Date"]) - pd.to_datetime('1970-01-01')).days

    # Prepare input data for the model
    input_data = pd.DataFrame({
        'reviews_per_month': [user_inputs["Reviews Per Month"]],
        'last_scraped': [last_scraped_days],  # Keep as numerical days since epoch
        'bedrooms': [user_inputs["Number of Bedrooms"]],
        'property_type': [user_inputs["Property Type"]],
        'latitude': [user_inputs["Latitude"]],
        'longitude': [user_inputs["Longitude"]],
        'price_per_bedroom': [user_inputs["Price Per Bedroom"]],
        'name': [int(user_inputs["Name (ID)"])]  # Ensure name is treated as integer
    })

    # Dynamic encoding for property type if needed
    df, property_type_encoder = encode_features(df, 'property_type')
    
    # Ensure that property type is encoded before prediction
    property_type_encoded = property_type_encoder.transform(input_data['property_type'])
    
    # Update input_data with encoded property type
    input_data['property_type'] = property_type_encoded

    # Scale input data if a scaler is provided
    if scaler is not None:
        input_data = scaler.transform(input_data)

    # Display button and predict when pressed
    if st.button("Predict Price"):
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"The predicted price is ${predicted_price:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

    # Display the sums of specified columns
    st.header("Data Summary")
    sums = calculate_sums(df)
    for key, value in sums.items():
        st.write(f"{key}: {value:.2f}")



# ------------------ Main Application ------------------ #
def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Select Page", 
        ["Home", "Geospatial Visualization", "Price Analysis", 
         "Availability Analysis", "Advanced Visualizations", "Predictive Analysis"]
    )

    # Load the appropriate data and model
    if selected_page == "Predictive Analysis":
        df = load_data(cleaned_data_path)  # Use cleaned data for prediction
        model = load_model(best_model_path)
    else:
        df = load_data(file_path)  # Use updated data for other pages

    # Page Navigation
    if selected_page == "Home":
        st.title("Airbnb Analysis")
        st.markdown("""
        ## Airbnb Analysis Project  
        **Project Creator:** Shubhangi Patil  
        **Skills:** Python, EDA, Data Visualization, Streamlit, and more  
        **Domain:** Travel, Property Management, and Tourism  
        **Problem Statement:** Analyze Airbnb data to uncover trends in pricing, availability, and property types.
        **GitHub:** [GitHub Repository](https://github.com/shubhangivspatil)
        """)

    elif selected_page == "Geospatial Visualization":
        geospatial_visualization(df)

    elif selected_page == "Price Analysis":
        price_analysis(df)

    elif selected_page == "Availability Analysis":
        availability_analysis(df)

    elif selected_page == "Advanced Visualizations":
        advanced_visualizations(df)

    elif selected_page == "Predictive Analysis":
        predictive_analysis(df, model)

if __name__ == "__main__":
    main()

