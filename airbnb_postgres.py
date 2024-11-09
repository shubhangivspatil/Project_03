import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sqlalchemy import create_engine
import plotly.express as px
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt

# Set up PostgreSQL connection
@st.cache_resource
def get_postgres_data():
    engine = create_engine('postgresql://postgres:admin@localhost:5432/Airbnb')
    query = "SELECT * FROM airbnb_data"  # Adjust the table name if different
    data = pd.read_sql(query, engine)
    return data

# Data Cleaning and Preparation
@st.cache_data
def preprocess_data(data):
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data.dropna(subset=['latitude', 'longitude', 'price'], inplace=True)
    return data

# Home Page
def home(data):
    st.title("Airbnb Analysis Dashboard")
    st.subheader("Project Title: Airbnb Analysis")

    st.markdown("""
        ### Skills Takeaway from This Project:
        - Python scripting, Data Preprocessing, Visualization, EDA, Streamlit, Postgres, PowerBI or Tableau

        ### Domain:
        - Travel Industry, Property Management, and Tourism

        ### Problem Statement:
        This project aims to analyze Airbnb data using MongoDB Atlas, perform data cleaning and preparation, develop interactive geospatial visualizations, and create dynamic plots to gain insights into pricing variations, availability patterns, and location-based trends.

        **User Scenario:**
        If you're planning a trip, this tool allows you to explore Airbnb listings in a specific country. You can filter for preferred room types, analyze average prices, find top-rated options, and assess availability to make informed decisions.
    """)

    st.image("https://image.shutterstock.com/z/stock-photo-beautiful-airbnb-apartment-1246322746.jpg", caption="Airbnb Listings Analysis", use_column_width=True)

    st.markdown("---")
    st.markdown("""
        **Created by [Shubhangi Patil](https://github.com/shubhangivspatil/Project_03)**
    """)



# Country Analysis Page
def country_analysis(data):
    st.header("Analyze Airbnb Listings by Country")

    # Select a country to analyze
    countries = data['country'].unique()
    selected_country = st.selectbox("Select Country", countries)

    # Filter data for the selected country
    country_data = data[data['country'] == selected_country]

    # City selection based on selected country
    cities = country_data['city'].unique()
    selected_city = st.selectbox("Select City", cities)

    # Filter data for the selected city
    city_data = country_data[country_data['city'] == selected_city]

    st.subheader(f"Overview of Airbnb Listings in {selected_city}, {selected_country}")

    # Display the average price in the selected city
    avg_price = city_data['price'].mean()
    st.write(f"**Average Price**: ${avg_price:.2f}")

    # Display highly rated listings
    top_rated = city_data[city_data['review_score_rating'] >= 4.5]
    st.write(f"**Number of Highly Rated Listings**: {top_rated.shape[0]}")

    # Show the top 5 highly rated listings by name and rating
    st.write("**Top 5 Highly Rated Listings**:")
    for _, row in top_rated.nlargest(5, 'review_score_rating').iterrows():
        st.write(f"- {row['listing_name']} (Rating: {row['review_score_rating']}/5)")

    # Display the most common property types
    property_types = city_data.filter(regex='property_type_').sum().sort_values(ascending=False).head(5)
    st.write("**Most Common Property Types:**")
    for prop, count in property_types.items():
        st.write(f"- {prop.replace('property_type_', '').replace('_', ' ')}: {count} listings")

    # Display average availability statistics
    avg_availability = city_data[['availability_30', 'availability_60', 'availability_90', 'availability_365']].mean()
    st.write("**Average Availability:**")
    st.write(f"- Next 30 Days: {avg_availability['availability_30']:.1f} days")
    st.write(f"- Next 60 Days: {avg_availability['availability_60']:.1f} days")
    st.write(f"- Next 90 Days: {avg_availability['availability_90']:.1f} days")
    st.write(f"- Next Year: {avg_availability['availability_365']:.1f} days")

    # Display the average availability category distribution
    availability_counts = city_data['availability_category'].value_counts(normalize=True) * 100
    st.write("**Availability Category Distribution:**")
    for category, percentage in availability_counts.items():
        st.write(f"- {category}: {percentage:.1f}%")

    # Interactive chart: Average price by availability category
    st.write("### Price Comparison by Availability Category")
    price_by_category = city_data.groupby('availability_category')['price'].mean().reset_index()
    fig = px.bar(price_by_category, x='availability_category', y='price',
                 title="Average Price by Availability Category",
                 labels={'price': 'Average Price', 'availability_category': 'Availability Category'})
    st.plotly_chart(fig)

    # Interactive chart: Price distribution
    st.write("### Price Distribution")
    fig = px.histogram(city_data, x='price', nbins=20,
                       title="Price Distribution of Listings",
                       labels={'price': 'Price (in $)'})
    st.plotly_chart(fig)

# Main geomaps function for geospatial analysis
def geomaps(data):
    """
    Displays an interactive geospatial map of Airbnb listings using Folium and Streamlit.
    Allows the user to select a country, city, adjust heatmap settings, and filter data by price and room type.
    
    Args:
        data (pd.DataFrame): The Airbnb data to be visualized.
    """
    st.header("Geospatial Analysis of Airbnb Listings")

    # Check if essential columns exist
    required_columns = ['country', 'city', 'latitude', 'longitude', 'price', 'listing_name', 'availability_category', 'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return

    # Country selection dropdown (use a unique key to prevent duplication)
    countries = data['country'].unique()
    selected_country = st.selectbox("Select Country for Map View", countries, key="country_selectbox")

    # Filter data for the selected country
    country_data = data[data['country'] == selected_country]

    # City selection dropdown
    cities = country_data['city'].unique()
    selected_city = st.selectbox("Select City for Map View", cities, key="city_selectbox")

    # Filter data for the selected city
    city_data = country_data[country_data['city'] == selected_city]

    # Check for empty data (this could be a reason why the map isn't showing)
    if city_data.empty:
        st.error(f"No data available for {selected_city}.")
        return

    # Map center based on average coordinates of the listings
    st.subheader(f"Map of Airbnb Listings in {selected_city}, {selected_country}")
    m = folium.Map(location=[city_data['latitude'].mean(), city_data['longitude'].mean()], zoom_start=12)

    # Sidebar for heatmap settings
    st.sidebar.subheader("Heat Map Settings")
    heat_radius = st.sidebar.slider("Heat Map Radius", min_value=5, max_value=30, value=10, key="heat_radius")
    heat_blur = st.sidebar.slider("Heat Map Blur", min_value=5, max_value=30, value=15, key="heat_blur")

    # Heatmap Data Preparation
    heat_data = city_data[['latitude', 'longitude']].values
    HeatMap(heat_data, radius=heat_radius, blur=heat_blur, max_zoom=1).add_to(m)

    # Sidebar for price range filter
    st.sidebar.subheader("Filter Options")
    min_price, max_price = st.sidebar.slider(
        "Select Price Range", 
        min_value=int(city_data['price'].min()), 
        max_value=int(city_data['price'].max()), 
        value=(int(city_data['price'].min()), int(city_data['price'].max())),
        key="price_range"
    )

    # Apply price filter to the data
    filtered_data = city_data[(city_data['price'] >= min_price) & (city_data['price'] <= max_price)]

    # Room Type Filtering based on binary columns
    room_type_options = ['Entire home/apt', 'Private room', 'Shared room']
    selected_room_type = st.sidebar.selectbox("Select Room Type", room_type_options)

    # Filter based on selected room type
    if selected_room_type == 'Entire home/apt':
        filtered_data = filtered_data[filtered_data['room_type_Entire home/apt'] == 1]
    elif selected_room_type == 'Private room':
        filtered_data = filtered_data[filtered_data['room_type_Private room'] == 1]
    elif selected_room_type == 'Shared room':
        filtered_data = filtered_data[filtered_data['room_type_Shared room'] == 1]

    # Marker clustering and color coding based on price
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in filtered_data.iterrows():
        price = row['price']
        color = 'green' if price < 100 else 'orange' if price < 300 else 'red'
        tooltip_text = f"{row['listing_name']} - ${price} - {row['availability_category']}"
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['listing_name']}: ${price} ({row['availability_category']})",
            tooltip=tooltip_text,
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(marker_cluster)

    # Add a legend for price categories with correctly formatted HTML
    legend_html = '''
    <div style="position: fixed; bottom: 10px; left: 10px; width: 160px; height: 110px; background-color: white; 
                border: 2px solid black; padding: 10px; font-size: 12px;">
        <b>Price Legend</b><br>
        <i style="background-color:green; width: 12px; height: 12px; display: inline-block;"></i> < $100<br>
        <i style="background-color:orange; width: 12px; height: 12px; display: inline-block;"></i> $100-$300<br>
        <i style="background-color:red; width: 12px; height: 12px; display: inline-block;"></i> > $300
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Layer Control to toggle Heat Map and Marker Cluster
    folium.LayerControl().add_to(m)

    # Display the interactive map in Streamlit
    st_folium(m, width=700, height=500)

    # Optionally, add advanced filters like availability category, room type, etc.
    st.sidebar.subheader("Advanced Filters")
    st.subheader(f"Filtered Listings in {selected_city}, {selected_country}")
    st.write(f"**Number of Listings after Filter:** {filtered_data.shape[0]}")
    st.write(f"**Price Range:** ${min_price} - ${max_price}")

    st.write("**Filtered Listings (Top 5):**")
    st.write(filtered_data.head())

    # Advanced visualization: Price Distribution
    st.subheader(f"Price Distribution in {selected_city}, {selected_country}")
    fig = px.histogram(filtered_data, x="price", nbins=30, title=f"Price Distribution for {selected_city}, {selected_country}")
    st.plotly_chart(fig, use_container_width=True)

def availability_category_analysis(data):
    st.header("Availability Analysis by Category and City")

    # Country selection
    countries = data['country'].unique()
    selected_country = st.selectbox("Select Country for Availability Category Analysis", countries)

    # Filter the data for the selected country
    country_data = data[data['country'] == selected_country]

    # City selection
    cities = country_data['city'].unique()
    selected_city = st.selectbox("Select City for Availability Category Analysis", cities)

    # Filter the data for the selected city
    city_data = country_data[country_data['city'] == selected_city]

    # Availability Category distribution
    availability_categories = city_data['availability_category'].value_counts()

    # Display the availability category distribution as a bar chart
    st.subheader("Availability Category Distribution")
    st.bar_chart(availability_categories)

    # Calculate the average availability for each category
    st.subheader("Average Availability by Category")
    avg_availability = city_data.groupby('availability_category')[['availability_30', 'availability_60', 'availability_90', 'availability_365']].mean()

    # Display the average availability
    st.write(f"**Average Availability for {selected_city}, {selected_country}:**")
    st.dataframe(avg_availability)

    # Optional: More details about each category
    st.subheader("Details on Availability Categories")
    for category in availability_categories.index:
        category_data = city_data[city_data['availability_category'] == category]
        st.write(f"**{category} Category:**")
        st.write(f"- Total Listings: {len(category_data)}")
        st.write(f"- Avg. Availability in next 30 days: {category_data['availability_30'].mean():.1f} days")
        st.write(f"- Avg. Availability in next 60 days: {category_data['availability_60'].mean():.1f} days")
        st.write(f"- Avg. Availability in next 90 days: {category_data['availability_90'].mean():.1f} days")
        st.write(f"- Avg. Availability in next 365 days: {category_data['availability_365'].mean():.1f} days")
        st.write("---")

    # Add some overall insights or charts
    st.subheader("Insight: Availability Trend Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    city_data.groupby('availability_category').agg({
        'availability_30': 'mean',
        'availability_60': 'mean',
        'availability_90': 'mean',
        'availability_365': 'mean'
    }).plot(kind='bar', ax=ax)
    ax.set_title(f'Average Availability by Category in {selected_city}, {selected_country}')
    ax.set_ylabel('Average Days Available')
    ax.set_xlabel('Availability Category')
    st.pyplot(fig)


def host_analysis(data):
    st.header("Host Analysis")

    # Country selection
    countries = data['country'].unique()
    selected_country = st.selectbox("Select Country for Host Analysis", countries)

    # Filter data by country
    country_data = data[data['country'] == selected_country]

    # City selection
    cities = country_data['city'].unique()
    selected_city = st.selectbox("Select City for Host Analysis", cities)

    # Filter data by city
    city_data = country_data[country_data['city'] == selected_city]

    # Number of unique hosts
    unique_hosts = city_data['host_id'].nunique()
    st.write(f"**Number of Unique Hosts in {selected_city}, {selected_country}:** {unique_hosts}")

    # Average host rating
    avg_host_rating = city_data.groupby('host_id')['review_score_rating'].mean().mean()
    st.write(f"**Average Host Rating in {selected_city}, {selected_country}:** {avg_host_rating:.2f}")

    # Top 10 hosts with the most listings
    listings_per_host = city_data.groupby('host_id').size().sort_values(ascending=False).head(10)
    st.write("**Top 10 Hosts with the Most Listings:**")
    st.write(listings_per_host)

    # Bar chart for top 10 hosts
    fig = px.bar(listings_per_host, x=listings_per_host.index, y=listings_per_host.values,
                 labels={'x': 'Host ID', 'y': 'Number of Listings'}, title=f"Top 10 Hosts in {selected_city}, {selected_country} by Listings")
    st.plotly_chart(fig, use_container_width=True)

    # Superhost analysis
    superhost_data = city_data[city_data['host_is_superhost'] == 'TRUE']
    non_superhost_data = city_data[city_data['host_is_superhost'] == 'FALSE']

    superhost_count = superhost_data['host_id'].nunique()
    non_superhost_count = non_superhost_data['host_id'].nunique()

    st.write(f"**Number of Superhosts in {selected_city}, {selected_country}:** {superhost_count}")
    st.write(f"**Number of Non-Superhosts in {selected_city}, {selected_country}:** {non_superhost_count}")

    avg_superhost_listings = superhost_data.groupby('host_id').size().mean()
    avg_non_superhost_listings = non_superhost_data.groupby('host_id').size().mean()

    st.write(f"**Average Number of Listings for Superhosts:** {avg_superhost_listings:.2f}")
    st.write(f"**Average Number of Listings for Non-Superhosts:** {avg_non_superhost_listings:.2f}")

    # Host location map
    st.subheader(f"Host Locations in {selected_city}, {selected_country}")
    host_location_data = city_data.groupby(['latitude', 'longitude']).size().reset_index(name='listing_count')

    m = folium.Map(location=[city_data['latitude'].mean(), city_data['longitude'].mean()], zoom_start=10)
    for _, row in host_location_data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Host has {row['listing_count']} listings",
            tooltip=f"Latitude: {row['latitude']}, Longitude: {row['longitude']}"
        ).add_to(m)

    st_folium(m, width=700, height=500)

    # Display additional host details: host_name and superhost status
    host_details = city_data[['host_id', 'host_name', 'host_is_superhost', 'host_location', 'host_total_listings_count']].drop_duplicates()
    st.write("**Host Details**")
    st.write(host_details)
# Advanced Visualization Page
def advanced_visualizations(data):
    st.header("Advanced Visualizations")

    # Country selection
    countries = data['country'].unique()
    selected_country = st.selectbox("Select Country for Price Distribution", countries)

    # Create a new column to represent room type
    room_type_columns = ['room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room']
    room_type_map = {
        'room_type_Entire home/apt': 'Entire home/apt',
        'room_type_Private room': 'Private room',
        'room_type_Shared room': 'Shared room'
    }

    # Find the active room type based on the room type columns
    for col in room_type_columns:
        if col in data.columns:
            data['room_type'] = data[room_type_columns].idxmax(axis=1).apply(lambda x: room_type_map.get(x, 'Unknown'))

    # Room type selection
    room_types = ['Entire home/apt', 'Private room', 'Shared room']
    selected_room_type = st.selectbox("Select Room Type", room_types)

    # Filter data based on selected country and room type
    filtered_data = data[(data['country'] == selected_country) & (data['room_type'] == selected_room_type)]

    # Price Distribution Histogram
    fig = px.histogram(filtered_data, 
                        x="price", 
                        nbins=30, 
                        title=f"Price Distribution for {selected_room_type} in {selected_country}",
                        labels={"price": "Price (USD)"})
    fig.update_layout(
        xaxis_title="Price (USD)", 
        yaxis_title="Count",
        template="plotly_dark",  
        bargap=0.1  
    )
    st.plotly_chart(fig, use_container_width=True)

    # Price vs. Number of Reviews Scatter Plot
    fig = px.scatter(filtered_data, 
                     x="number_of_reviews", 
                     y="price", 
                     color="price",  # Apply color to differentiate by price
                     color_continuous_scale="Cividis", 
                     title="Price vs. Number of Reviews",
                     labels={"price": "Price (USD)", "number_of_reviews": "Number of Reviews"})
    fig.update_traces(marker=dict(size=8, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title="Number of Reviews",
        yaxis_title="Price (USD)",
        template="plotly_dark",  
        plot_bgcolor='rgba(0,0,0,0)'  
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cancellation Policy Distribution Bar Plot
    cancellation_policy_counts = data[data['country'] == selected_country]['cancellation_policy'].value_counts()
    fig = px.bar(cancellation_policy_counts, 
                 x=cancellation_policy_counts.index, 
                 y=cancellation_policy_counts.values, 
                 color=cancellation_policy_counts.index, 
                 title=f"Cancellation Policy Distribution in {selected_country}",
                 labels={"x": "Cancellation Policy", "y": "Number of Listings"})
    fig.update_layout(
        template="plotly_dark",  
        xaxis_title="Cancellation Policy",
        yaxis_title="Number of Listings",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Price Distribution by Room Type (Box Plot)
    fig = px.box(data[data['country'] == selected_country], 
                 x="room_type", 
                 y="price", 
                 title="Price Distribution by Room Type",
                 labels={"room_type": "Room Type", "price": "Price (USD)"})
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Room Type",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Average Price by Room Type (Bar Plot)
    avg_price_by_room = data[data['country'] == selected_country].groupby('room_type')['price'].mean().reset_index()
    fig = px.bar(avg_price_by_room, 
                 x="room_type", 
                 y="price", 
                 color="price", 
                 color_continuous_scale="Blues", 
                 title="Average Price by Room Type",
                 labels={"room_type": "Room Type", "price": "Average Price (USD)"})
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Room Type",
        yaxis_title="Average Price (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Price vs. Availability (Scatter Plot)
    fig = px.scatter(filtered_data, 
                     x="availability_365", 
                     y="price", 
                     color="price", 
                     color_continuous_scale="Inferno", 
                     title="Price vs. Availability (365 Days)",
                     labels={"availability_365": "Availability (Days)", "price": "Price (USD)"})
    fig.update_traces(marker=dict(size=8, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title="Availability (Days)",
        yaxis_title="Price (USD)",
        template="plotly_dark",  
        plot_bgcolor='rgba(0,0,0,0)'  
    )
    st.plotly_chart(fig, use_container_width=True)

    # Geographical Distribution of Listings (Map)
    st.subheader(f"Geographical Distribution of Listings in {selected_country}")
    map_data = data[data['country'] == selected_country]
    m = folium.Map(location=[map_data['latitude'].mean(), map_data['longitude'].mean()], zoom_start=10)
    for _, row in map_data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Price: ${row['price']}, Reviews: {row['number_of_reviews']}",
            tooltip=f"Latitude: {row['latitude']}, Longitude: {row['longitude']}"
        ).add_to(m)
    st_folium(m, width=700, height=500)

    # Price vs. Location (Scatter Plot)
    fig = px.scatter(map_data, 
                     x="longitude", 
                     y="latitude", 
                     color="price", 
                     size="price", 
                     hover_data=["price", "number_of_reviews"], 
                     title="Price vs. Location (Latitude vs Longitude)",
                     labels={"latitude": "Latitude", "longitude": "Longitude"})
    fig.update_layout(
        template="plotly_dark",  
        xaxis_title="Longitude",
        yaxis_title="Latitude"
    )
    st.plotly_chart(fig, use_container_width=True)
    #Review-Level Filtering Page
def review_level_filtering(data):
    st.header("Review-Level Filtering")

    # Step 1: Country Selection
    countries = data['country'].unique()
    selected_country = st.selectbox("Select Country for Review Filtering", countries)

    country_data = data[data['country'] == selected_country]

    # Step 2: City Selection
    cities = country_data['city'].unique()
    selected_city = st.selectbox("Select City", cities)

    city_data = country_data[country_data['city'] == selected_city]

    st.subheader(f"Listings in {selected_city}, {selected_country}")
    st.write(f"**Number of Listings:** {city_data.shape[0]}")
    st.write(f"**Average Price:** ${city_data['price'].mean():.2f}")

    # Step 3: Display Average Price by Room Type
    room_types = [col for col in data.columns if 'room_type_' in col]  # Dynamically find room_type columns
    city_data_grouped = city_data.groupby(room_types)['price'].mean().reset_index()  # Reset index to make 'room_type' usable in Plotly
    st.write("**Average Price by Room Type:**")
    st.write(city_data_grouped)

    # Step 4: Review Ratings Analysis (display various ratings)
    st.subheader("Review Scores")

    review_columns = [
        "review_score_rating", 
        "review_score_accuracy", 
        "review_score_cleanliness", 
        "review_score_location", 
        "review_score_value"
    ]
    
    # Calculate the average review scores for each category
    review_avg = city_data[review_columns].mean()
    
    # Display average review scores
    st.write("**Average Review Scores:**")
    st.write(f"**Rating:** {review_avg['review_score_rating']:.2f}")
    st.write(f"**Accuracy:** {review_avg['review_score_accuracy']:.2f}")
    st.write(f"**Cleanliness:** {review_avg['review_score_cleanliness']:.2f}")
    st.write(f"**Location:** {review_avg['review_score_location']:.2f}")
    st.write(f"**Value:** {review_avg['review_score_value']:.2f}")
    
    # Step 5: Plotting Average Review Ratings by Room Type
    st.subheader("Review Scores by Room Type")
    
    # Create a table of average reviews by room type
    review_by_room = city_data.groupby(room_types)[review_columns].mean().reset_index()  # Reset index to make 'room_type' usable in Plotly
    
    st.write("**Average Review Scores by Room Type:**")
    st.write(review_by_room)
    
    # Optionally, add visualization (bar charts for review categories)
    
    # Plot review scores by room type
    for review_column in review_columns:
        fig = px.bar(review_by_room, 
                     x=review_by_room[room_types[0]],  # Use the first room_type column
                     y=review_column, 
                     title=f"Average {review_column.replace('_', ' ').title()} by Room Type",
                     labels={"room_type": "Room Type", review_column: review_column.replace('_', ' ').title()})
        st.plotly_chart(fig, use_container_width=True)
    
    # Step 6: Display Number of Reviews
    st.subheader("Number of Reviews")
    st.write(f"**Average Number of Reviews:** {city_data['number_of_reviews'].mean():.0f}")
    st.write(f"**Total Number of Reviews across Listings:** {city_data['number_of_reviews'].sum():,.0f}")
    
    # Optionally, create a histogram of number of reviews
    fig = px.histogram(city_data, 
                        x="number_of_reviews", 
                        nbins=20, 
                        title="Distribution of Number of Reviews", 
                        labels={"number_of_reviews": "Number of Reviews"})
    st.plotly_chart(fig, use_container_width=True)
# Define Streamlit Pages
PAGES = {
    "Home": home,
    "Country Analysis": country_analysis,
    "Geospatial Analysis": geomaps,
    "Seasonal Availability Analysis": availability_category_analysis,
    "Host Analysis": host_analysis,
    "Advanced Visualizations": advanced_visualizations,
    "Review-Level Filtering Page": review_level_filtering
}

# Set up sidebar for page navigation
def main():
    st.sidebar.title("Airbnb Analysis Dashboard")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    data = get_postgres_data()  # Load the data at the start of the app
    data = preprocess_data(data)  # Preprocess the data
    page(data)
   
    

if __name__ == "__main__":
    main()
