import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

@st.cache_data  # Updated from st.cache to st.cache_data
def load_data():
    file_path = r'D:\GUVI_Projects\My_Projects\updated_cleaned_airbnb_data.csv'
    data = pd.read_csv(file_path)
    
    # Convert 'last_scraped' to datetime if it exists in the DataFrame
    if 'last_scraped' in data.columns:
        data['last_scraped'] = pd.to_datetime(data['last_scraped'], dayfirst=True)
    
    # Clean the 'price' column to remove dollar signs and convert to float
    if 'price' in data.columns:
        data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Save the processed DataFrame to a new CSV file for later use in Power BI
    output_file_path = r'D:\GUVI_Projects\My_Projects\Final_Airbnb.csv'
    data.to_csv(output_file_path, index=False)
    
    return data

df = load_data()

st.title('Airbnb Analysis')
st.sidebar.title("Filters")

# Sidebar for selecting neighbourhood and date range
if 'neighbourhood_cleansed' in df.columns:
    selected_neighbourhood = st.sidebar.selectbox("Select Neighbourhood", options=np.unique(df['neighbourhood_cleansed']))
    filtered_data = df[df['neighbourhood_cleansed'] == selected_neighbourhood]
else:
    filtered_data = df

if 'last_scraped' in df.columns:
    start_date, end_date = st.sidebar.select_slider(
        "Select the date range",
        options=pd.date_range(start=df['last_scraped'].min(), end=df['last_scraped'].max(), freq='D'),
        value=(df['last_scraped'].min(), df['last_scraped'].max())
    )
    filtered_data = filtered_data[(filtered_data['last_scraped'] >= start_date) & (filtered_data['last_scraped'] <= end_date)]

# Price Analysis
st.header("Price Analysis")
price_fig = px.histogram(filtered_data, x="price", color="property_type", barmode="overlay")
st.plotly_chart(price_fig)

# Seasonal Availability
st.header("Seasonal Availability")
if 'last_scraped' in filtered_data.columns:
    availability_fig = px.line(filtered_data, x='last_scraped', y='availability_365', color='property_type')
    st.plotly_chart(availability_fig)

# Map of Listings
st.header("Map of Listings")
if 'number_of_reviews' in filtered_data.columns:
    map_fig = px.scatter_mapbox(
        filtered_data,
        lat="latitude",
        lon="longitude",
        color="price",
        size="number_of_reviews",
        hover_name="name",
        mapbox_style="open-street-map",
        zoom=10
    )
    st.plotly_chart(map_fig)

# Insights for Selected Neighbourhood
st.header(f"Insights for {selected_neighbourhood}")
neighbourhood_data = filtered_data[filtered_data['neighbourhood_cleansed'] == selected_neighbourhood]
insight_fig = px.bar(neighbourhood_data, x='neighbourhood_cleansed', y='price', color='availability_365')
st.plotly_chart(insight_fig)

# Key Insights from the Entire Dataset
st.header("Key Insights from the Entire Dataset")
average_price = np.mean(df['price'])
st.write(f"Overall average price: ${average_price:.2f}")

# Property Type Distribution
st.subheader("Property Type Distribution")
property_types = df['property_type'].value_counts()
fig_property_types = px.bar(x=property_types.index, y=property_types.values, labels={'x': 'Property Type', 'y': 'Count'})
st.plotly_chart(fig_property_types)

# Top 5 Most Common Neighborhoods
st.subheader("Top 5 Most Common Neighborhoods")
common_neighborhoods = df['neighbourhood_cleansed'].value_counts().nlargest(5)
fig_neighborhoods = px.bar(x=common_neighborhoods.index, y=common_neighborhoods.values, labels={'x': 'Neighborhood', 'y': 'Count'})
st.plotly_chart(fig_neighborhoods)

# Availability Patterns
st.subheader("Availability Patterns")
avg_availability = df.groupby('neighbourhood_cleansed')['availability_365'].mean().sort_values(ascending=False).nlargest(5)
fig_availability = px.bar(avg_availability, labels={'value': 'Average Days Available', 'index': 'Neighborhood'})
st.plotly_chart(fig_availability)

# Review Patterns Across Listings
st.subheader("Review Patterns Across Listings")
avg_reviews = df['number_of_reviews'].mean()
max_reviews = df['number_of_reviews'].max()
st.write(f"Average number of reviews per listing: {avg_reviews:.1f}")
st.write(f"Maximum number of reviews received by a single listing: {max_reviews}")

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(filtered_data)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv',
)




