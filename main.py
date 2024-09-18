import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
import streamlit as st
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

sns.set_style('whitegrid')

# Streamlit interface
st.title("Netflix Clustering Analysis")
st.write("Upload your Netflix dataset in CSV format.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Rename columns for better readability
    df.rename(columns={
        "director": "Director", 
        "country": "Country", 
        "cast": "Cast", 
        "date_added": "Date_Added",
        "rating": "Rating",
        "duration": "Duration",
        "show_id": "Show_ID",
        "type": "Video_Type",
        "title": "Title", 
        "release_year": "Year_Release",
        "listed_in": "Genres",
        "description": "Description"}, 
        inplace=True)

    # Fill missing values
    df['Duration'].fillna(df['Duration'].mode()[0], inplace=True)
    df['Rating'].fillna(df['Rating'].mode()[0], inplace=True)
    df.fillna({'Director': 'Unknown', 'Cast': 'Unknown', 'Country': 'Unknown'}, inplace=True)

    # Data Preprocessing for K-Prototypes
    movie_df = df[df['Video_Type'] == 'Movie']
    tv_show_df = df[df['Video_Type'] == 'TV Show']

    # Process 'Duration'
    movie_df['Duration'] = movie_df['Duration'].str.extract('(\d+)').astype(int)
    tv_show_df['Duration'] = tv_show_df['Duration'].str.extract('(\d+)').astype(int)

    # Explode 'Country'
    movie_df['Country'] = movie_df['Country'].str.split(', ')
    movie_df = movie_df.explode('Country').reset_index(drop=True)
    tv_show_df['Country'] = tv_show_df['Country'].str.split(', ')
    tv_show_df = tv_show_df.explode('Country').reset_index(drop=True)

    # Convert 'Country' to categorical codes
    movie_df['Country'] = movie_df['Country'].astype('category').cat.codes
    tv_show_df['Country'] = tv_show_df['Country'].astype('category').cat.codes

    # Prepare data for K-Prototypes clustering
    movie_data_kproto = movie_df[['Duration', 'Country']]
    tv_show_data_kproto = tv_show_df[['Duration', 'Country']]

    # Clustering parameters
    movie_clusters = st.slider("Select number of clusters for Movies:", 2, 10, 3)
    tv_show_clusters = st.slider("Select number of clusters for TV Shows:", 2, 10, 4)

    # K-Prototypes for Movies
    kproto_movie = KPrototypes(n_clusters=movie_clusters, init='Cao', n_init=5, verbose=1)
    movie_clusters_result = kproto_movie.fit_predict(movie_data_kproto, categorical=[1])
    movie_df['Cluster'] = movie_clusters_result

    # K-Prototypes for TV Shows
    kproto_tv_show = KPrototypes(n_clusters=tv_show_clusters, init='Cao', n_init=5, verbose=1)
    tv_show_clusters_result = kproto_tv_show.fit_predict(tv_show_data_kproto, categorical=[1])
    tv_show_df['Cluster'] = tv_show_clusters_result

    # Visualizations
    st.subheader('Movie Clusters')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=movie_df, x='Duration', y='Country', hue='Cluster', palette='viridis', s=100)
    plt.title('Movie DataFrame Clusters')
    plt.xlabel('Duration')
    plt.ylabel('Country')
    plt.legend(title='Cluster')
    st.pyplot(plt)

    st.subheader('TV Show Clusters')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tv_show_df, x='Duration', y='Country', hue='Cluster', palette='viridis', s=100)
    plt.title('TV Show DataFrame Clusters')
    plt.xlabel('Duration')
    plt.ylabel('Country')
    plt.legend(title='Cluster')
    st.pyplot(plt)

    # Calculate metrics
    movie_silhouette_score = silhouette_score(movie_data_kproto, movie_clusters_result)
    tv_show_silhouette_score = silhouette_score(tv_show_data_kproto, tv_show_clusters_result)

    movie_davies_bouldin_score = davies_bouldin_score(movie_data_kproto, movie_clusters_result)
    tv_show_davies_bouldin_score = davies_bouldin_score(tv_show_data_kproto, tv_show_clusters_result)

    movie_calinski_harabasz_score = calinski_harabasz_score(movie_data_kproto, movie_clusters_result)
    tv_show_calinski_harabasz_score = calinski_harabasz_score(tv_show_data_kproto, tv_show_clusters_result)

    # Create a DataFrame for score comparison
    scores = {
        "Metric": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
        "Movies": [movie_silhouette_score, movie_davies_bouldin_score, movie_calinski_harabasz_score],
        "TV Shows": [tv_show_silhouette_score, tv_show_davies_bouldin_score, tv_show_calinski_harabasz_score]
    }
    
    scores_df = pd.DataFrame(scores)

    # Display the comparison table
    st.subheader("Clustering Performance Comparison")
    st.table(scores_df)

    # Genre Analysis
    # Movie genre analysis
    if 'Cluster' in movie_df.columns and 'Genres' in movie_df.columns:
        genre_cluster_counts = movie_df.groupby(['Cluster', 'Genres']).size().reset_index(name='Count')
        genre_cluster_counts = genre_cluster_counts.sort_values(by='Count', ascending=False)
        top_genres_per_cluster = genre_cluster_counts.groupby('Cluster').head(10)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=top_genres_per_cluster, x='Count', y='Genres', hue='Cluster', palette='viridis')
        plt.title('Top Genres by Number of Movies in Each Cluster')
        plt.xlabel('Number of Movies')
        plt.ylabel('Genres')
        plt.legend(title='Cluster')
        st.pyplot(plt)

    # TV show genre analysis
    if 'Cluster' in tv_show_df.columns and 'Genres' in tv_show_df.columns:
        tv_genre_cluster_counts = tv_show_df.groupby(['Cluster', 'Genres']).size().reset_index(name='Count')
        tv_genre_cluster_counts = tv_genre_cluster_counts.sort_values(by='Count', ascending=False)
        top_tv_genres_per_cluster = tv_genre_cluster_counts.groupby('Cluster').head(10)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=top_tv_genres_per_cluster, x='Count', y='Genres', hue='Cluster', palette='viridis')
        plt.title('Top Genres by Number of TV Shows in Each Cluster')
        plt.xlabel('Number of TV Shows')
        plt.ylabel('Genres')
        plt.legend(title='Cluster')
        st.pyplot(plt)
