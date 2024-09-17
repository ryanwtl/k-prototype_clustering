# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from ipywidgets import interact
from sklearn.metrics import silhouette_score

sns.set_style('whitegrid')

# Load the dataset
df = pd.read_csv(r'netflix.csv')

# Rename columns for better readability
df.rename(columns={
    "director": "Director", 
    "country": "Country", 
    "cast":"Cast", 
    "date_added":"Date_Added",
    "rating":"Rating",
    "duration":"Duration",
    "show_id":"Show_ID",
    "type":"Video_Type",
    "title":"Title", 
    "release_year":"Year_Release",
    "listed_in":"Genres",
    "description":"Description"}, 
    inplace=True)

# Fill missing values in Duration and Rating columns
df['Duration'].fillna(df['Duration'].mode()[0], inplace=True)
df['Rating'].fillna(df['Rating'].mode()[0], inplace=True)

# Fill missing values for Country, Director, Cast with 'Unknown'
df.fillna({'Director': 'Unknown', 'Cast': 'Unknown', 'Country': 'Unknown'}, inplace=True)

# Data Preprocessing for K-Prototypes
movie_df = df[df['Video_Type'] == 'Movie']
tv_show_df = df[df['Video_Type'] == 'TV Show']

# Convert 'Duration' to integer by extracting the numeric part
movie_df['Duration'] = movie_df['Duration'].str.extract('(\d+)').astype(int)
tv_show_df['Duration'] = tv_show_df['Duration'].str.extract('(\d+)').astype(int)

# Explode 'Country' into separate rows and convert to categorical codes
movie_df['Country'] = movie_df['Country'].str.split(', ')
movie_df = movie_df.explode('Country').reset_index(drop=True)
tv_show_df['Country'] = tv_show_df['Country'].str.split(', ')
tv_show_df = tv_show_df.explode('Country').reset_index(drop=True)

# Convert 'Country' to categorical codes for clustering
movie_df['Country'] = movie_df['Country'].astype('category').cat.codes
tv_show_df['Country'] = tv_show_df['Country'].astype('category').cat.codes

# Prepare data for K-Prototypes clustering
movie_data_kproto = movie_df[['Duration', 'Country']]
tv_show_data_kproto = tv_show_df[['Duration', 'Country']]

# Interactive function for adjusting clustering parameters and visualizing results
def kproto_clustering(movie_clusters=3, tv_show_clusters=4):
    # K-Prototypes for Movies
    kproto_movie = KPrototypes(n_clusters=movie_clusters, init='Cao', n_init=5, verbose=1)
    movie_clusters_result = kproto_movie.fit_predict(movie_data_kproto, categorical=[1])
    movie_df['Cluster'] = movie_clusters_result
    
    # K-Prototypes for TV Shows
    kproto_tv_show = KPrototypes(n_clusters=tv_show_clusters, init='Cao', n_init=5, verbose=1)
    tv_show_clusters_result = kproto_tv_show.fit_predict(tv_show_data_kproto, categorical=[1])
    tv_show_df['Cluster'] = tv_show_clusters_result

    # Visualization for Movies
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=movie_df, x='Duration', y='Country', hue='Cluster', palette='viridis', s=100)
    plt.title('Movie DataFrame Clusters')
    plt.xlabel('Duration')
    plt.ylabel('Country')
    plt.legend(title='Cluster')
    plt.show()
    
    # Visualization for TV Shows
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tv_show_df, x='Duration', y='Country', hue='Cluster', palette='viridis', s=100)
    plt.title('TV Show DataFrame Clusters')
    plt.xlabel('Duration')
    plt.ylabel('Country')
    plt.legend(title='Cluster')
    plt.show()

    # Calculate Silhouette Scores
    movie_silhouette = silhouette_score(movie_data_kproto, movie_clusters_result)
    tv_show_silhouette = silhouette_score(tv_show_data_kproto, tv_show_clusters_result)

    print(f"Silhouette Score for Movie Clusters: {movie_silhouette}")
    print(f"Silhouette Score for TV Show Clusters: {tv_show_silhouette}")

# Use interactive widget to control the number of clusters
interact(kproto_clustering, movie_clusters=(2, 10), tv_show_clusters=(2, 10))

# --------------------- Genre Analysis for Movies ---------------------
# Make sure 'Cluster' and 'Genres' columns exist
if 'Cluster' in movie_df.columns and 'Genres' in movie_df.columns:
    # Group by Cluster and Genres to count movies for each genre within each cluster
    genre_cluster_counts = movie_df.groupby(['Cluster', 'Genres']).size().reset_index(name='Count')

    # Sort the data by the count of movies
    genre_cluster_counts = genre_cluster_counts.sort_values(by='Count', ascending=False)

    # Filter to keep only the top N genres in each cluster (for example, top 10 genres per cluster)
    top_genres_per_cluster = genre_cluster_counts.groupby('Cluster').head(10)

    # Plot the popularity of genres within each cluster using a bar plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=top_genres_per_cluster, x='Count', y='Genres', hue='Cluster', palette='viridis')
    plt.title('Top Genres by Number of Movies in Each Cluster')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genres')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()
else:
    print("Either 'Cluster' or 'Genres' column is missing in movie_df")

# --------------------- Genre Analysis for TV Shows ---------------------
# Make sure 'Cluster' and 'Genres' columns exist
if 'Cluster' in tv_show_df.columns and 'Genres' in tv_show_df.columns:
    # Group by Cluster and Genres to count TV shows for each genre within each cluster
    tv_genre_cluster_counts = tv_show_df.groupby(['Cluster', 'Genres']).size().reset_index(name='Count')

    # Sort the data by the count of TV shows
    tv_genre_cluster_counts = tv_genre_cluster_counts.sort_values(by='Count', ascending=False)

    # Filter to keep only the top N genres in each cluster (for example, top 10 genres per cluster)
    top_tv_genres_per_cluster = tv_genre_cluster_counts.groupby('Cluster').head(10)

    # Plot the popularity of genres within each cluster using a bar plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=top_tv_genres_per_cluster, x='Count', y='Genres', hue='Cluster', palette='viridis')
    plt.title('Top Genres by Number of TV Shows in Each Cluster')
    plt.xlabel('Number of TV Shows')
    plt.ylabel('Genres')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()
else:
    print("Either 'Cluster' or 'Genres' column is missing in tv_show_df")
