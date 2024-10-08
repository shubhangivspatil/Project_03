import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset function
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

# Geospatial Visualization
def geospatial_visualization(df):
    st.title("Geospatial Visualization of Airbnb Listings")
    st.header("Map of Listings")
    
    # Basic Scatter Map
    if 'number_of_reviews' in df.columns:
        map_fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="price",
                                     size="number_of_reviews", hover_name="name",
                                     mapbox_style="open-street-map", zoom=10,
                                     title="Distribution of Airbnb Listings")
        st.plotly_chart(map_fig)
        st.write("This map visualizes the distribution of Airbnb listings. The size of the markers indicates the number of reviews, while the color represents the price range. Areas with larger markers signify more popular listings.")

    # Geospatial Distribution of Prices
    st.header("Geospatial Distribution of Prices")
    price_map_fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="price",
                                       size_max=15, hover_name="name",
                                       mapbox_style="open-street-map", zoom=10,
                                       title="Geospatial Distribution of Prices")
    st.plotly_chart(price_map_fig)
    st.write("The map shows how prices vary across different geographical locations. Areas with darker colors indicate higher average prices, providing insights into expensive neighborhoods.")

    # Hot Spot Map
    st.header("Hot Spot Map of Listings")
    heat_map_data = df[['latitude', 'longitude', 'price']].dropna()
    heat_map_data = heat_map_data[heat_map_data['price'] > 0]  # Filter out non-positive prices
    heat_map = folium.Map(location=[heat_map_data['latitude'].mean(), heat_map_data['longitude'].mean()], zoom_start=12)
    HeatMap(data=heat_map_data[['latitude', 'longitude']], radius=15, max_zoom=13).add_to(heat_map)

    st.write("This hot spot map highlights areas with high concentrations of Airbnb listings. Darker areas indicate more listings, which can suggest popular locations for travelers.")
    
    # Render the folium map
    folium_static(heat_map)

# Availability Analysis
def availability_analysis(df):
    st.title("Availability Analysis of Airbnb Listings")
    
    # Seasonal Availability Analysis
    st.header("Availability Analysis by Season")
    df['season'] = pd.cut(df['availability_365'], bins=[-1, 90, 180, 270, 365], 
                          labels=['Low', 'Moderate', 'High', 'Very High'])
    
    season_counts = df['season'].value_counts().sort_index()
    bar_fig = px.bar(season_counts, x=season_counts.index, y=season_counts.values,
                      title='Availability by Season', labels={'x': 'Season', 'y': 'Count'})
    st.plotly_chart(bar_fig)
    st.write("This bar chart illustrates the distribution of listings across different availability seasons. A higher count in 'Very High' indicates a large number of listings available year-round.")

    # Seasonal Booking Patterns
    df['month'] = pd.to_datetime(df['last_scraped']).dt.month
    monthly_availability = df.groupby('month')['availability_365'].mean().reset_index()
    line_fig = px.line(monthly_availability, x='month', y='availability_365',
                        title='Average Monthly Availability', labels={'month': 'Month', 'availability_365': 'Average Availability'})
    st.plotly_chart(line_fig)
    st.write("This line chart displays the average availability of listings throughout the year. Peaks during certain months may indicate seasonal demand variations.")

# Price Analysis
def price_analysis(df):
    st.title("Price Analysis of Airbnb Listings")
    st.header("Average Price Trends Over Time")
    df['last_scraped'] = pd.to_datetime(df['last_scraped'])
    time_series_fig = px.line(df, x="last_scraped", y="price", title="Average Price Trends Over Time",
                               labels={"last_scraped": "Date", "price": "Average Price"})
    st.plotly_chart(time_series_fig)
    st.write("This line chart illustrates how average prices have changed over time. Notable spikes may indicate increased demand during certain periods, such as holidays or events.")

    st.header("Price Distribution by Property Type")
    property_type_filter = st.selectbox("Select Property Type", options=df['property_type'].unique())
    filtered_df = df[df['property_type'] == property_type_filter]
    box_fig = px.box(filtered_df, x="property_type", y="price", title="Price Distribution by Property Type")
    st.plotly_chart(box_fig)
    st.write("The box plot displays the distribution of prices for the selected property type. The central line represents the median price, while the boxes show the interquartile range. Outliers may suggest unique, high-priced listings.")

# Advanced Visualizations
def advanced_visualizations(df):
    st.title("Advanced Visualizations of Airbnb Listings")
    st.header("Pair Plot of Price and Other Variables")
    pair_plot_fig = sns.pairplot(df[['price', 'number_of_reviews', 'availability_365', 'latitude', 'longitude']])
    st.pyplot(pair_plot_fig)
    st.write("This pair plot visualizes relationships between price and other key variables. Look for trends or clusters that might indicate pricing strategies based on reviews or location.")

    st.header("Sunburst Chart of Property Types and Prices")
    sunburst_fig = px.sunburst(df, path=['neighbourhood_cleansed', 'property_type'], values='price',
                                title="Sunburst Chart of Property Types and Prices")
    st.plotly_chart(sunburst_fig)
    st.write("The sunburst chart shows the hierarchical relationship between neighborhoods and property types concerning their average prices. This visualization can help identify which property types dominate specific neighborhoods.")

    st.header("Choropleth Map of Average Prices by Neighbourhood")
    choropleth_fig = px.choropleth(df, locations='neighbourhood_cleansed', locationmode='country names',
                                    color='price', hover_name='neighbourhood_cleansed',
                                    title="Choropleth Map of Average Prices by Neighbourhood")
    st.plotly_chart(choropleth_fig)
    st.write("This choropleth map visualizes average prices by neighborhood. Darker shades indicate higher average prices, which may correlate with factors like location and amenities.")

    st.header("Price Distribution with Box and Violin Plot")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='property_type', y='price', data=df, palette="Set2")
    sns.violinplot(x='property_type', y='price', data=df, inner='quartile', color='lightgray')
    plt.title("Box and Violin Plot of Price Distribution by Property Type")
    st.pyplot(plt)
    st.write("This combined box and violin plot visualizes price distributions across property types. The violin plot provides a density estimate of the price distribution, highlighting variations within types.")

# Main app
def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Select Page", ["Home", "Geospatial Visualization", "Price Analysis", 
                                                       "Availability Analysis", "Advanced Visualizations"])

    file_path = r'D:\GUVI_Projects\My_Projects\updated_cleaned_airbnb_data.csv'
    df = load_data(file_path)

    # Home Page
    if selected_page == "Home":
        st.title("Airbnb Analysis")
        st.write("Welcome to the Airbnb Analysis Project")
        
        st.markdown("""
        ## Project Title
        **Airbnb Analysis**

        ## Skills Takeaway From This Project
        Python scripting, Data Preprocessing, Visualization,
        EDA, Streamlit, MongoDB, PowerBI or Tableau 

        ## Domain
        Travel Industry, Property Management, and Tourism 

        ## Problem Statement
        This project aims to analyze Airbnb data using MongoDB Atlas, perform data cleaning and preparation, develop interactive geospatial visualizations, and create dynamic plots to gain insights into pricing variations, availability patterns, and location-based trends.
        
        ## Creator
        **Shubhangi Patil**

        ## Project
        Data Science

        ## GitHub Link
        [GitHub Repository](https://github.com/shubhangivspatil)
        """)

    elif selected_page == "Geospatial Visualization":
        geospatial_visualization(df)

    elif selected_page == "Price Analysis":
        price_analysis(df)

    elif selected_page == "Availability Analysis":
        availability_analysis(df)

    elif selected_page == "Advanced Visualizations":
        advanced_visualizations(df)

if __name__ == "__main__":
    main()
