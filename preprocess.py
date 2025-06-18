import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data():
    """
    Preprocess the book data and save the results for faster app loading
    """
    # Path to data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Load data
    print("Loading datasets...")
    books = pd.read_csv(os.path.join(data_dir, 'Books.csv'), low_memory=False, dtype={'Year-Of-Publication': 'str'})
    ratings = pd.read_csv(os.path.join(data_dir, 'Ratings.csv'))
    users = pd.read_csv(os.path.join(data_dir, 'Users.csv'))
    
    # Clean data
    books = books.dropna(subset=['Book-Title', 'Book-Author'])
    ratings = ratings.dropna()
    
    print(f"After cleaning - Books: {books.shape[0]}, Ratings: {ratings.shape[0]}, Users: {users.shape[0]}")
    
    # Create popularity based recommendation
    print("Creating popularity based recommendation data...")
    ratings_with_name = ratings.merge(books, on='ISBN')
    
    # Count ratings per book
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
    
    # Calculate average rating per book
    avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
    
    # Create popularity dataframe
    popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    
    # Filter by minimum number of ratings and sort by rating
    popular_books = popularity_df[popularity_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False)
    
    # Merge with books to get additional details
    top_books = popular_books.merge(books, on='Book-Title').drop_duplicates('Book-Title')
    selected_columns = ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating', 'Year-Of-Publication']
    top_books = top_books[selected_columns]
    
    print(f"Generated {len(top_books)} popular books")
      # Create collaborative filtering data
    print("Creating collaborative filtering data...")
    # More aggressive filtering to reduce memory usage further
    # Filter very active users (users with more ratings)
    user_counts = ratings_with_name.groupby('User-ID').size()
    active_users = user_counts[user_counts >= 200].index  # Even higher threshold
    
    filtered_ratings = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]
    
    # Filter very popular books
    book_counts = filtered_ratings.groupby('Book-Title').size()
    popular_books_collab = book_counts[book_counts >= 100].index  # Higher threshold
    
    final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(popular_books_collab)]
    
    # Limit the number of books to prevent memory issues - make it even smaller
    top_books_for_collab = final_ratings.groupby('Book-Title').size().sort_values(ascending=False).head(500).index
    final_ratings = final_ratings[final_ratings['Book-Title'].isin(top_books_for_collab)]
    
    print(f"Using {len(final_ratings)} ratings for {len(top_books_for_collab)} books from {final_ratings['User-ID'].nunique()} users")
    
    # Create pivot table
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    
    print(f"Created pivot table with {pt.shape[0]} books and {pt.shape[1]} users")
      # Calculate similarity scores
    print("Calculating similarity scores (this might take a while)...")
    try:
        similarity_scores = cosine_similarity(pt)
        print(f"Generated similarity scores of shape {similarity_scores.shape}")
    except MemoryError:
        print("Memory error occurred. Trying with even smaller dataset...")
        # Further reduce the dataset
        top_books_for_collab = final_ratings.groupby('Book-Title').size().sort_values(ascending=False).head(300).index
        final_ratings = final_ratings[final_ratings['Book-Title'].isin(top_books_for_collab)]
        
        pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
        pt.fillna(0, inplace=True)
        
        similarity_scores = cosine_similarity(pt)
        print(f"Generated similarity scores of shape {similarity_scores.shape} with reduced dataset")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data - save only pivot table, not similarity scores to save memory
    print("Saving processed data...")
    with open(os.path.join(output_dir, 'top_books.pkl'), 'wb') as f:
        pickle.dump(top_books, f)
    
    with open(os.path.join(output_dir, 'pt.pkl'), 'wb') as f:
        pickle.dump(pt, f)
    
    # Don't save similarity scores - we'll calculate them on demand
    print("Skipping similarity scores to save memory - will calculate on demand")
    
    print("Preprocessing completed!")
    print(f"Files saved in: {output_dir}")
    print("Note: Similarity scores will be calculated on-demand for better memory usage")

if __name__ == "__main__":
    preprocess_data()
