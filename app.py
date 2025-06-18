import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
from PIL import Image
from io import BytesIO
import streamlit_lottie
import pickle
import re
import time

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'reading_history' not in st.session_state:
    st.session_state.reading_history = []

# Custom CSS
def local_css():
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        bg_color = "#1E1E1E"
        card_bg = "#2D2D2D"
        text_color = "#FFFFFF"
        secondary_text = "#B0B0B0"
        accent_color = "#3B82F6"
    else:
        bg_color = "#f5f5f5"
        card_bg = "white"
        text_color = "#1E3A8A"
        secondary_text = "#6B7280"
        accent_color = "#1E3A8A"
    
    st.markdown(f"""
    <style>
        .main {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: {text_color};
        }}
        .stButton>button {{
            background-color: {accent_color};
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #2563EB;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        .book-card {{
            background-color: {card_bg};
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: all 0.3s ease;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }}
        .book-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }}
        .book-title {{
            font-weight: bold;
            font-size: 18px;
            color: {text_color};
            margin-bottom: 8px;
        }}
        .book-author {{
            font-style: italic;
            color: {secondary_text};
            margin-bottom: 10px;
        }}
        .rating {{
            color: #FCD34D;
            font-size: 16px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, {accent_color}, #3B82F6);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        .genre-tag {{
            background-color: {accent_color};
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin: 2px;
            display: inline-block;
        }}
        .favorite-btn {{
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .favorite-btn:hover {{
            transform: scale(1.2);
        }}
        .search-box {{
            background-color: {card_bg};
            border: 2px solid {accent_color};
            border-radius: 25px;
            padding: 10px 20px;
            margin: 10px 0;
        }}
        .recommendation-score {{
            background: linear-gradient(90deg, #10B981, #059669);
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .analytics-card {{
            background: {card_bg};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid {accent_color};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
    </style>
    """, unsafe_allow_html=True)

local_css()

# Load animation
@st.cache_data
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Functions to load and process data
@st.cache_data
def load_data():
    # Get the directory of the current script
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Load data with optimized data types and memory settings
    books = pd.read_csv(os.path.join(data_dir, 'Books.csv'), low_memory=False, dtype={'Year-Of-Publication': 'str'})
    ratings = pd.read_csv(os.path.join(data_dir, 'Ratings.csv'))
    users = pd.read_csv(os.path.join(data_dir, 'Users.csv'))
    
    # Clean and optimize data
    books = books.dropna(subset=['Book-Title', 'Book-Author'])
    ratings = ratings.dropna()
    
    return books, ratings, users

@st.cache_data
def load_preprocessed_data():
    """Load preprocessed data if available, otherwise preprocess from scratch"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    try:
        # Only load the smaller files first
        with open(os.path.join(model_dir, 'top_books.pkl'), 'rb') as f:
            top_books = pickle.load(f)
        
        st.sidebar.success("Loaded preprocessed popular books data")
        return top_books, None, None
        
    except (FileNotFoundError, EOFError):
        st.sidebar.warning("Preprocessed data not found. Run preprocess.py first for faster loading.")
        return None, None, None

@st.cache_data
def load_collaborative_filtering_models():
    """Load collaborative filtering models separately to manage memory better"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    try:
        # Load pivot table
        with open(os.path.join(model_dir, 'pt.pkl'), 'rb') as f:
            pt = pickle.load(f)
        
        st.sidebar.success("Loaded collaborative filtering pivot table")
        return pt, None  # Don't load similarity scores yet
        
    except (FileNotFoundError, EOFError, MemoryError) as e:
        st.sidebar.warning(f"Could not load collaborative filtering models: {str(e)}")
        return None, None

@st.cache_data
def prepare_popularity_based_data(books, ratings, top_books=None):
    """Get popular books data"""
    if top_books is not None:
        return top_books
        
    # If no preprocessed data, calculate from scratch
    # Merge ratings with books
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
    
    return top_books[selected_columns]

@st.cache_data
def prepare_collaborative_filtering_data(books, ratings, pt=None, similarity_scores=None):
    """Get collaborative filtering data"""
    if pt is not None and similarity_scores is not None:
        return pt, similarity_scores, books
        
    # If no preprocessed data, calculate from scratch with memory optimization
    st.info("Processing collaborative filtering data. This may take a moment...")
    
    # Merge ratings with books
    ratings_with_name = ratings.merge(books, on='ISBN')
    
    # More aggressive filtering to reduce memory usage
    # Filter very active users (users with more ratings)
    user_counts = ratings_with_name.groupby('User-ID').size()
    active_users = user_counts[user_counts >= 100].index  # Increased threshold
    
    filtered_ratings = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]
    
    # Filter very popular books
    book_counts = filtered_ratings.groupby('Book-Title').size()
    popular_books = book_counts[book_counts >= 50].index  # Increased threshold
    
    final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(popular_books)]
    
    # Limit the number of books to prevent memory issues
    top_books = final_ratings.groupby('Book-Title').size().sort_values(ascending=False).head(1000).index
    final_ratings = final_ratings[final_ratings['Book-Title'].isin(top_books)]
    
    st.info(f"Using {len(final_ratings)} ratings for {len(top_books)} books from {final_ratings['User-ID'].nunique()} users")
    
    # Create pivot table
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    
    # Calculate similarity scores with memory optimization
    try:
        similarity_scores = cosine_similarity(pt)
    except MemoryError:
        st.error("Memory error occurred. Please run preprocess.py first to create optimized model files.")
        st.stop()
    
    return pt, similarity_scores, books

@st.cache_data
def get_book_recommendations(book_name, pt, similarity_scores, books, n=5):
    # Find index of the book
    if book_name not in pt.index:
        return pd.DataFrame()
        
    index = np.where(pt.index == book_name)[0][0]
    
    # If similarity_scores is None, calculate on-demand for this book only
    if similarity_scores is None:
        # Calculate similarity for just this book
        book_vector = pt.iloc[index].values.reshape(1, -1)
        similarities = cosine_similarity(book_vector, pt.values)[0]
        similar_items = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)[1:n+1]
    else:
        # Use precomputed similarities
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:n+1]
    
    # Create dataframe with recommended books
    recommended_books = []
    for i in similar_items:
        book_title = pt.index[i[0]]
        temp_df = books[books['Book-Title'] == book_title]
        if not temp_df.empty:
            book_details = {
                'Book-Title': book_title,
                'Book-Author': temp_df['Book-Author'].values[0],
                'Image-URL-M': temp_df['Image-URL-M'].values[0],
                'Year-Of-Publication': temp_df['Year-Of-Publication'].values[0],
                'Similarity-Score': i[1]
            }
            recommended_books.append(book_details)
    
    return pd.DataFrame(recommended_books)

@st.cache_data
def get_book_details(book_title, books):
    return books[books['Book-Title'] == book_title]

# Function to display book cards
def display_book_card(book_title, book_author, image_url, rating=None, num_ratings=None, year=None, similarity_score=None, isbn=None):
    with st.container():
        st.markdown(f'<div class="book-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Try to get better cover image from APIs
            better_cover_url = get_book_cover_from_apis(book_title, book_author, isbn)
            
            # Use the better cover URL if available, otherwise fall back to original
            final_image_url = better_cover_url if better_cover_url and "placeholder" not in better_cover_url else image_url
            
            try:
                if final_image_url and final_image_url.startswith('http'):
                    response = requests.get(final_image_url, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=150)
                else:
                    st.image("https://via.placeholder.com/150x200?text=No+Cover+Available", width=150)
            except Exception as e:
                # Show original URL if API fails
                try:
                    if image_url and image_url.startswith('http'):
                        response = requests.get(image_url, timeout=5)
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x200?text=No+Cover+Available", width=150)
                except:
                    st.image("https://via.placeholder.com/150x200?text=No+Cover+Available", width=150)
        
        with col2:
            st.markdown(f'<p class="book-title">{book_title}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="book-author">by {book_author}</p>', unsafe_allow_html=True)
            
            if year:
                st.text(f"Published: {year}")
                
            if rating:
                st.markdown(f'<p class="rating">Rating: {"★" * int(rating)} ({rating:.1f}/5)</p>', unsafe_allow_html=True)
            
            if num_ratings:
                st.text(f"Based on {num_ratings} ratings")
                
            if similarity_score:
                st.progress(similarity_score, text=f"Match Score: {similarity_score*100:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Function to display book cards with new features
def display_enhanced_book_card(book_title, book_author, image_url, rating=None, num_ratings=None, year=None, similarity_score=None, isbn=None, context=""):
    with st.container():
        st.markdown(f'<div class="book-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 3, 1])
        
        # Create unique keys by combining hash with context and timestamp
        unique_id = f"{hash(book_title)}_{hash(book_author)}_{context}_{time.time()}"
        
        with col1:
            # Try to get better cover image from APIs
            better_cover_url = get_book_cover_from_apis(book_title, book_author, isbn)
            
            # Use the better cover URL if available, otherwise fall back to original
            final_image_url = better_cover_url if better_cover_url and "placeholder" not in better_cover_url else image_url
            
            try:
                if final_image_url and final_image_url.startswith('http'):
                    response = requests.get(final_image_url, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=150)
                else:
                    st.image("https://via.placeholder.com/150x200?text=No+Cover+Available", width=150)
            except Exception as e:
                # Show original URL if API fails
                try:
                    if image_url and image_url.startswith('http'):
                        response = requests.get(image_url, timeout=5)
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x200?text=No+Cover+Available", width=150)
                except:
                    st.image("https://via.placeholder.com/150x200?text=No+Cover+Available", width=150)
        
        with col2:
            # Title and Author
            st.markdown(f'<p class="book-title">{book_title}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="book-author">by {book_author}</p>', unsafe_allow_html=True)
            
            # Book details
            if year:
                st.text(f"📅 Published: {year}")
                
            if rating:
                stars = "⭐" * int(rating)
                st.markdown(f'<p class="rating">{stars} ({rating:.1f}/5)</p>', unsafe_allow_html=True)
            
            if num_ratings:
                st.text(f"👥 {num_ratings:,} ratings")
                
            if similarity_score:
                st.progress(similarity_score, text=f"📊 Match Score: {similarity_score*100:.1f}%")
        
        with col3:
            # Favorite button
            is_favorite = book_title in st.session_state.favorites
            if st.button("❤️" if is_favorite else "🤍", key=f"fav_{unique_id}", help="Add to favorites"):
                if is_favorite:
                    st.session_state.favorites.remove(book_title)
                    st.success("Removed from favorites!")
                else:
                    st.session_state.favorites.append(book_title)
                    st.success("Added to favorites!")
                time.sleep(0.5)
                st.rerun()
            
            # User rating section
            user_rating = st.selectbox(
                "Rate:", 
                [0, 1, 2, 3, 4, 5], 
                index=st.session_state.user_ratings.get(book_title, 0),
                key=f"rating_{unique_id}",
                format_func=lambda x: "⭐" if x == 0 else f"{'⭐' * x}"
            )
            if user_rating > 0:
                st.session_state.user_ratings[book_title] = user_rating
            
            # Read button
            if st.button("📚", key=f"read_{unique_id}", help="Mark as read"):
                if book_title not in [item['title'] for item in st.session_state.reading_history]:
                    st.session_state.reading_history.append({
                        'title': book_title,
                        'author': book_author,
                        'date_read': pd.Timestamp.now().strftime('%Y-%m-%d')
                    })
                    st.success("Added to reading history!")
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Book Cover API Functions
@st.cache_data
def get_book_cover_from_apis(title, author, isbn=None):
    """
    Fetch book cover from multiple APIs with fallback system
    Returns the best available cover image URL
    """
    cover_url = None
    
    # Method 1: Try Open Library Covers API with ISBN
    if isbn and isbn.strip():
        cover_url = get_openlibrary_cover(isbn)
        if cover_url:
            return cover_url
    
    # Method 2: Try Google Books API with title and author
    cover_url = get_google_books_cover(title, author)
    if cover_url:
        return cover_url
    
    # Method 3: Try Open Library Search API
    cover_url = get_openlibrary_search_cover(title, author)
    if cover_url:
        return cover_url
    
    # Fallback: Return placeholder
    return "https://via.placeholder.com/150x200?text=No+Cover+Available"

@st.cache_data
def get_openlibrary_cover(isbn, size='M'):
    """
    Get book cover from Open Library Covers API using ISBN
    Size options: S (small), M (medium), L (large)
    """
    try:
        # Clean ISBN (remove hyphens, spaces)
        clean_isbn = re.sub(r'[^0-9X]', '', str(isbn).upper())
        
        if not clean_isbn:
            return None
            
        # Try both ISBN formats
        for isbn_format in [clean_isbn, isbn]:
            url = f"https://covers.openlibrary.org/b/isbn/{isbn_format}-{size}.jpg?default=false"
            
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return url
                
        return None
    except Exception:
        return None

@st.cache_data
def get_google_books_cover(title, author):
    """
    Get book cover from Google Books API
    """
    try:
        # Clean and prepare search query
        query_parts = []
        if title:
            query_parts.append(f'intitle:"{title}"')
        if author:
            query_parts.append(f'inauthor:"{author}"')
            
        if not query_parts:
            return None
            
        query = '+'.join(query_parts)
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                book = data['items'][0]
                if 'volumeInfo' in book and 'imageLinks' in book['volumeInfo']:
                    image_links = book['volumeInfo']['imageLinks']
                    # Prefer higher quality images
                    for quality in ['large', 'medium', 'small', 'thumbnail', 'smallThumbnail']:
                        if quality in image_links:
                            # Convert HTTP to HTTPS for security
                            cover_url = image_links[quality].replace('http://', 'https://')
                            return cover_url
        
        return None
    except Exception:
        return None

@st.cache_data
def get_openlibrary_search_cover(title, author):
    """
    Get book cover from Open Library Search API
    """
    try:
        # Prepare search query
        query_parts = []
        if title:
            query_parts.append(f'title:"{title}"')
        if author:
            query_parts.append(f'author:"{author}"')
            
        if not query_parts:
            return None
            
        query = ' AND '.join(query_parts)
        url = f"https://openlibrary.org/search.json?q={query}&limit=1"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            if 'docs' in data and len(data['docs']) > 0:
                book = data['docs'][0]
                
                # Try to get cover using various IDs
                for id_type, id_key in [('id', 'cover_i'), ('isbn', 'isbn'), ('lccn', 'lccn')]:
                    if id_key in book and book[id_key]:
                        if isinstance(book[id_key], list):
                            book_id = book[id_key][0]
                        else:
                            book_id = book[id_key]
                        
                        if id_type == 'id':
                            cover_url = f"https://covers.openlibrary.org/b/id/{book_id}-M.jpg"
                        else:
                            cover_url = f"https://covers.openlibrary.org/b/{id_type}/{book_id}-M.jpg"
                            
                        # Test if cover exists
                        test_response = requests.head(cover_url, timeout=5)
                        if test_response.status_code == 200:
                            return cover_url
        
        return None
    except Exception:
        return None

@st.cache_data
def test_image_url(url):
    """
    Test if an image URL is valid and accessible
    """
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except:
        return False

# Analytics and Statistics Functions
def display_user_analytics():
    """Display user reading analytics"""
    st.subheader("📊 Your Reading Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="stat-card">
            <h3>{len(st.session_state.favorites)}</h3>
            <p>Favorite Books</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="stat-card">
            <h3>{len(st.session_state.reading_history)}</h3>
            <p>Books Read</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        rated_books = len([r for r in st.session_state.user_ratings.values() if r > 0])
        st.markdown(f'''
        <div class="stat-card">
            <h3>{rated_books}</h3>
            <p>Books Rated</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        avg_rating = np.mean([r for r in st.session_state.user_ratings.values() if r > 0]) if st.session_state.user_ratings else 0
        st.markdown(f'''
        <div class="stat-card">
            <h3>{avg_rating:.1f}</h3>
            <p>Avg Rating</p>
        </div>
        ''', unsafe_allow_html=True)

def display_reading_history():
    """Display user's reading history"""
    if st.session_state.reading_history:
        st.subheader("📚 Your Reading History")
        df = pd.DataFrame(st.session_state.reading_history)
        df['date_read'] = pd.to_datetime(df['date_read'])
        df = df.sort_values('date_read', ascending=False)
        
        for _, book in df.iterrows():
            with st.container():
                st.markdown(f'''
                <div class="analytics-card">
                    <strong>{book['title']}</strong><br>
                    <em>by {book['author']}</em><br>
                    <small>📅 Read on: {book['date_read'].strftime('%B %d, %Y')}</small>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.info("No books in your reading history yet. Start adding books!")

def get_advanced_recommendations(books, ratings, user_ratings, n_recommendations=10):
    """Get recommendations based on user's ratings"""
    if not user_ratings:
        return pd.DataFrame()
    
    # Create user profile based on ratings
    rated_books = list(user_ratings.keys())
    user_preferences = pd.DataFrame({
        'Book-Title': rated_books,
        'User-Rating': [user_ratings[book] for book in rated_books]
    })
    
    # Merge with book data to get authors
    user_books = user_preferences.merge(books[['Book-Title', 'Book-Author']], on='Book-Title', how='left')
    
    # Get favorite authors (books rated 4 or 5)
    favorite_authors = user_books[user_books['User-Rating'] >= 4]['Book-Author'].value_counts()
    
    if len(favorite_authors) > 0:
        # Recommend books by favorite authors that user hasn't rated
        author_recommendations = books[
            (books['Book-Author'].isin(favorite_authors.index)) &
            (~books['Book-Title'].isin(rated_books))
        ].head(n_recommendations)
        
        return author_recommendations
    
    return pd.DataFrame()

# Additional Recommendation Functions
@st.cache_data
def get_author_based_recommendations(selected_book, books, n=5):
    """Get recommendations based on the same author"""
    book_details = books[books['Book-Title'] == selected_book]
    if book_details.empty:
        return pd.DataFrame()
    
    author = book_details['Book-Author'].iloc[0]
    
    # Find other books by the same author
    author_books = books[
        (books['Book-Author'] == author) & 
        (books['Book-Title'] != selected_book)
    ].head(n)
    
    return author_books

@st.cache_data
def get_popular_recommendations(books, ratings, n=5):
    """Get recommendations based on popularity (high ratings and many reviews)"""
    # Calculate book popularity
    ratings_with_books = ratings.merge(books, on='ISBN')
    
    # Group by book title and calculate stats
    book_stats = ratings_with_books.groupby('Book-Title').agg({
        'Book-Rating': ['mean', 'count'],
        'Book-Author': 'first',
        'Image-URL-M': 'first',
        'Year-Of-Publication': 'first',
        'ISBN': 'first'
    }).reset_index()
    
    # Flatten column names
    book_stats.columns = ['Book-Title', 'avg_rating', 'num_ratings', 'Book-Author', 'Image-URL-M', 'Year-Of-Publication', 'ISBN']
    
    # Filter books with at least 100 ratings and sort by rating
    popular_books = book_stats[book_stats['num_ratings'] >= 100].sort_values('avg_rating', ascending=False)
    
    return popular_books.head(n)

@st.cache_data
def get_year_based_recommendations(selected_book, books, n=5):
    """Get recommendations based on publication year (books from similar time period)"""
    book_details = books[books['Book-Title'] == selected_book]
    if book_details.empty:
        return pd.DataFrame()
    
    year = book_details['Year-Of-Publication'].iloc[0]
    
    try:
        # Handle different year formats and clean the data
        year_str = str(year).strip()
        
        # Skip if year is not available or not numeric
        if not year_str or year_str.lower() in ['nan', 'none', '', '0']:
            # Fallback: return random books
            return books[books['Book-Title'] != selected_book].sample(n=min(n, len(books)-1))
        
        year_num = int(float(year_str))
        
        # Create a more robust filter for years
        # First, clean the Year-Of-Publication column
        books_clean = books.copy()
        books_clean['Year-Clean'] = books_clean['Year-Of-Publication'].astype(str).str.strip()
        
        # Filter out invalid years
        books_clean = books_clean[
            (books_clean['Year-Clean'] != '') &
            (books_clean['Year-Clean'] != 'nan') &
            (books_clean['Year-Clean'] != 'None') &
            (books_clean['Year-Clean'] != '0')
        ]
        
        # Convert to numeric, handling errors
        books_clean['Year-Numeric'] = pd.to_numeric(books_clean['Year-Clean'], errors='coerce')
        
        # Filter books within reasonable year range (1800-2030) and within 10 years of selected book
        similar_year_books = books_clean[
            (books_clean['Year-Numeric'].notna()) &
            (books_clean['Year-Numeric'] >= 1800) &
            (books_clean['Year-Numeric'] <= 2030) &
            (abs(books_clean['Year-Numeric'] - year_num) <= 10) &  # Increased range to 10 years
            (books_clean['Book-Title'] != selected_book)
        ].head(n)
        
        # If no books found with 10 years, try 20 years
        if similar_year_books.empty:
            similar_year_books = books_clean[
                (books_clean['Year-Numeric'].notna()) &
                (books_clean['Year-Numeric'] >= 1800) &
                (books_clean['Year-Numeric'] <= 2030) &
                (abs(books_clean['Year-Numeric'] - year_num) <= 20) &
                (books_clean['Book-Title'] != selected_book)
            ].head(n)
        
        # If still no books, return books from the same decade
        if similar_year_books.empty:
            decade = (year_num // 10) * 10
            similar_year_books = books_clean[
                (books_clean['Year-Numeric'].notna()) &
                (books_clean['Year-Numeric'] >= decade) &
                (books_clean['Year-Numeric'] < decade + 10) &
                (books_clean['Book-Title'] != selected_book)
            ].head(n)
        
        # Drop the temporary columns before returning
        if not similar_year_books.empty:
            similar_year_books = similar_year_books.drop(['Year-Clean', 'Year-Numeric'], axis=1)
        
        return similar_year_books
        
    except Exception as e:
        # If anything fails, return some random books as fallback
        st.warning(f"Could not filter by year ({str(e)}). Showing random books instead.")
        return books[books['Book-Title'] != selected_book].sample(n=min(n, len(books)-1))

@st.cache_data
def get_random_recommendations(books, n=5):
    """Get random book recommendations"""
    # Sample random books
    random_books = books.sample(n=min(n, len(books)))
    return random_books

@st.cache_data
def get_title_similarity_recommendations(selected_book, books, n=5):
    """Get recommendations based on title similarity (simple text matching)"""
    import re
    
    # Extract key words from the selected book title
    selected_words = set(re.findall(r'\b\w+\b', selected_book.lower()))
    
    # Remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    selected_words = selected_words - common_words
    
    if not selected_words:
        return pd.DataFrame()
    
    # Find books with similar words in title
    similar_books = []
    for _, book in books.iterrows():
        if book['Book-Title'] == selected_book:
            continue
            
        book_words = set(re.findall(r'\b\w+\b', book['Book-Title'].lower()))
        book_words = book_words - common_words
        
        # Calculate similarity as intersection over union
        if book_words:
            similarity = len(selected_words.intersection(book_words)) / len(selected_words.union(book_words))
            if similarity > 0.1:  # At least 10% similarity
                similar_books.append((book, similarity))
    
    # Sort by similarity and return top n
    similar_books.sort(key=lambda x: x[1], reverse=True)
    result_books = [book[0] for book in similar_books[:n]]
    
    return pd.DataFrame(result_books)

# Genre Analysis Function  
def analyze_genres(books):
    """Analyze book genres and trends"""
    st.subheader("📈 Dataset Analytics")
    
    # Publication year analysis
    books_clean = books.dropna(subset=['Year-Of-Publication'])
    books_clean['Year-Of-Publication'] = pd.to_numeric(books_clean['Year-Of-Publication'], errors='coerce')
    books_clean = books_clean.dropna(subset=['Year-Of-Publication'])
    
    # Year distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Books by Publication Decade**")
        books_clean['Decade'] = (books_clean['Year-Of-Publication'] // 10) * 10
        decade_counts = books_clean['Decade'].value_counts().sort_index()
        
        # Display as bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        decade_counts.plot(kind='bar', ax=ax, color='#1E3A8A')
        ax.set_title('Books Published by Decade')
        ax.set_xlabel('Decade')
        ax.set_ylabel('Number of Books')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**👥 Top Authors by Book Count**")
        author_counts = books['Book-Author'].value_counts().head(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        author_counts.plot(kind='barh', ax=ax, color='#3B82F6')
        ax.set_title('Most Prolific Authors')
        ax.set_xlabel('Number of Books')
        plt.tight_layout()
        st.pyplot(fig)
        
# Main application
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/book-shelf.png", width=100)
    st.sidebar.title("Navigation")
    
    # Load animation
    lottie_book = load_lottie_url('https://assets8.lottiefiles.com/private_files/lf30_LOw4AL.json')
    if lottie_book:
        streamlit_lottie.st_lottie(lottie_book, height=200, key="book_animation")
    
    page = st.sidebar.radio("Select Page", [
        "🏠 Home", 
        "📊 Analytics", 
        "⭐ Popular Books", 
        "🎯 Book Recommendations", 
        "❤️ My Favorites",
        "📚 Reading History"
    ])
    
    # Theme toggle
    st.sidebar.markdown("---")
    if st.sidebar.button("🌙 Toggle Dark/Light Mode"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()
    
    st.sidebar.markdown(f"**Current Theme:** {st.session_state.theme.title()}")
    
    # Load data
    try:
        books, ratings, users = load_data()
        st.sidebar.success(f"Loaded {books.shape[0]} books, {ratings.shape[0]} ratings")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
      # Home page
    if page == "🏠 Home":
        st.title("📚 Book Recommendation System")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Welcome to our **Enhanced Book Recommendation System**! This application helps you discover new books with:
            
            * 📊 **Analytics Dashboard** - Track your reading habits and preferences
            * ⭐ **Popular Books** - Books that are highly rated by many users
            * 🎯 **Smart Recommendations** - Books similar to ones you already enjoy
            * ❤️ **Personal Favorites** - Save and manage your favorite books
            * 📚 **Reading History** - Track books you've read
            * 🌙 **Dark/Light Themes** - Customize your experience
            
            ### Dataset Information
            """)
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Books", f"{books.shape[0]:,}")
            col_b.metric("Users", f"{users.shape[0]:,}")
            col_c.metric("Ratings", f"{ratings.shape[0]:,}")
            
            st.markdown("""
            ### 🚀 New Features
            
            - **🎯 Interactive Ratings**: Rate books and get personalized recommendations
            - **❤️ Favorites System**: Save books you love for easy access
            - **📊 Personal Analytics**: See your reading statistics and trends  
            - **🔍 Enhanced Search**: Advanced filtering and search capabilities
            - **📱 Better UI**: Modern design with improved mobile experience
            
            Navigate using the sidebar on the left to explore all features!
            """)
        
        with col2:
            st.image("https://img.icons8.com/clouds/400/000000/book-shelf.png", width=250)
            
            # Quick stats about user activity
            if st.session_state.favorites or st.session_state.reading_history:
                st.subheader("📈 Your Quick Stats")
                if st.session_state.favorites:
                    st.info(f"❤️ {len(st.session_state.favorites)} books in favorites")
                if st.session_state.reading_history:
                    st.info(f"📚 {len(st.session_state.reading_history)} books read")
    
    # Analytics page
    elif page == "📊 Analytics":
        st.title("📊 Reading Analytics")
        
        # Display user analytics
        display_user_analytics()
          # Dataset analytics
        analyze_genres(books)
        
        # Advanced recommendations based on user ratings
        if st.session_state.user_ratings:
            st.subheader("🎯 Recommendations Based on Your Ratings")
            advanced_recs = get_advanced_recommendations(books, ratings, st.session_state.user_ratings)
            
            if not advanced_recs.empty:
                st.write("Books by authors you rated highly:")
                for _, book in advanced_recs.head(5).iterrows():
                    display_enhanced_book_card(
                        book['Book-Title'],
                        book['Book-Author'],
                        book.get('Image-URL-M', ''),
                        year=book.get('Year-Of-Publication', ''),
                        isbn=book.get('ISBN', ''),
                        context="analytics"
                    )
            else:
                st.info("Rate some books to get personalized author recommendations!")
      # Favorites page
    elif page == "❤️ My Favorites":
        st.title("❤️ My Favorite Books")
        
        if st.session_state.favorites:
            st.write(f"You have {len(st.session_state.favorites)} favorite books:")
            
            # Add option to clear all favorites
            if st.button("🗑️ Clear All Favorites"):
                st.session_state.favorites = []
                st.success("All favorites cleared!")
                st.rerun()
            
            # Display favorite books
            for book_title in st.session_state.favorites:
                book_details = get_book_details(book_title, books)
                if not book_details.empty:
                    book = book_details.iloc[0]
                    display_enhanced_book_card(
                        book['Book-Title'],
                        book['Book-Author'],
                        book.get('Image-URL-M', ''),
                        year=book.get('Year-Of-Publication', ''),
                        isbn=book.get('ISBN', ''),
                        context="favorites"
                    )
                else:
                    st.warning(f"Could not find details for: {book_title}")
        else:
            st.info("No favorite books yet! Start adding books to your favorites by clicking the ❤️ button on book cards.")
    
    # Reading History page
    elif page == "📚 Reading History":
        st.title("📚 Your Reading History")
        
        if st.session_state.reading_history:
            # Add option to clear reading history
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Clear History"):
                    st.session_state.reading_history = []
                    st.success("Reading history cleared!")
                    st.rerun()
            
            display_reading_history()
        else:
            st.info("No reading history yet! Mark books as read by clicking the 📚 button on book cards.")      # Popular Books page
    elif page == "⭐ Popular Books":
        st.title("⭐ Popular Books")
        st.write("These books are highly rated by thousands of readers")
        
        # Try to load preprocessed data first
        top_books_pkl, _, _ = load_preprocessed_data()
        top_books = prepare_popularity_based_data(books, ratings, top_books_pkl)
        
        # Enhanced filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            min_ratings = st.slider("Minimum number of ratings", 50, 1000, 250, step=50)
        with col2:
            min_avg_rating = st.slider("Minimum average rating", 1.0, 5.0, 3.0, step=0.1)
        with col3:
            books_to_show = st.selectbox("Books to display", [10, 20, 30, 50], index=1)
        
        # Apply filters
        filtered_books = top_books[
            (top_books['num_ratings'] >= min_ratings) & 
            (top_books['avg_rating'] >= min_avg_rating)
        ].head(books_to_show)
        
        st.write(f"Showing {len(filtered_books)} books matching your criteria")
          # Display books in a grid
        num_cols = 2
        rows = len(filtered_books) // num_cols + (1 if len(filtered_books) % num_cols > 0 else 0)
        
        for i in range(rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < len(filtered_books):
                    book = filtered_books.iloc[idx]
                    with cols[j]:
                        display_enhanced_book_card(
                            book['Book-Title'],
                            book['Book-Author'],
                            book['Image-URL-M'],
                            book['avg_rating'],
                            book['num_ratings'],
                            book['Year-Of-Publication'],
                            context="popular_books"
                        )# Book Recommendations page
    elif page == "🎯 Book Recommendations":
        st.title("🎯 Smart Book Recommendations")
        st.write("Get personalized book recommendations using different algorithms")
        
        # Try to load preprocessed data first
        top_books_pkl, _, _ = load_preprocessed_data()
        
        # Load collaborative filtering models separately
        pt, similarity_scores = load_collaborative_filtering_models()
        
        # Book selection interface
        st.subheader("📚 Step 1: Select a Book")
        
        # Enhanced search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search by book title", placeholder="Enter part of a book title...")
        with col2:
            search_limit = st.selectbox("Max results", [10, 25, 50, 100], index=1)
        
        # Book selection logic
        selected_book = None
        available_books = books['Book-Title'].unique().tolist() if pt is None else sorted(pt.index.tolist())
        
        if search_term:
            matched_books = [book for book in available_books if search_term.lower() in book.lower()][:search_limit]
            if matched_books:
                selected_book = st.selectbox("Select a book", matched_books)
            else:
                st.warning("No books match your search term")
        else:
            # Show some popular books as options
            if st.button("🎲 Pick Random Book"):
                selected_book = np.random.choice(available_books)
                st.success(f"Selected: {selected_book}")
          # Show selected book details
        if selected_book:
            book_details = get_book_details(selected_book, books)
            
            if not book_details.empty:
                st.subheader("📖 Your Selected Book")
                book = book_details.iloc[0]
                display_enhanced_book_card(
                    book['Book-Title'],
                    book['Book-Author'],
                    book.get('Image-URL-M', ''),
                    year=book.get('Year-Of-Publication', ''),
                    isbn=book.get('ISBN', ''),
                    context="selected_book"
                )
                
                # Recommendation Type Selection
                st.subheader("🎯 Step 2: Choose Recommendation Type")
                st.write("Select how you want to get recommendations:")
                
                # Recommendation buttons in a grid
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    smart_recs = st.button("🤖 Smart AI Recommendations", 
                                         help="AI-powered collaborative filtering based on user preferences",
                                         use_container_width=True)
                    author_recs = st.button("👤 More by Same Author", 
                                          help="Find other books by this author",
                                          use_container_width=True)
                
                with col2:
                    popular_recs = st.button("⭐ Popular Books", 
                                           help="Currently trending and highly-rated books",
                                           use_container_width=True)
                    year_recs = st.button("📅 Books from Same Era", 
                                        help="Books published around the same time",
                                        use_container_width=True)
                
                with col3:
                    similar_title_recs = st.button("📝 Similar Titles", 
                                                  help="Books with similar title themes",
                                                  use_container_width=True)
                    random_recs = st.button("🎲 Random Discovery", 
                                          help="Surprise me with random great books!",
                                          use_container_width=True)
                
                # Settings
                col_set1, col_set2 = st.columns([3, 1])
                with col_set2:
                    num_recs = st.selectbox("Number of recommendations", [3, 5, 8, 10], index=1)
                
                # Process recommendations based on button clicked
                recommendations = pd.DataFrame()
                recommendation_title = ""
                
                if smart_recs:
                    # Prepare collaborative filtering data if needed
                    try:
                        if pt is not None:
                            recommendations = get_book_recommendations(selected_book, pt, None, books, n=num_recs)
                            recommendation_title = "🤖 AI-Powered Smart Recommendations"
                        else:
                            pt, similarity_scores, books = prepare_collaborative_filtering_data(books, ratings, pt, None)
                            recommendations = get_book_recommendations(selected_book, pt, similarity_scores, books, n=num_recs)
                            recommendation_title = "🤖 AI-Powered Smart Recommendations"
                    except Exception as e:
                        st.error(f"Error with smart recommendations: {e}")
                        st.info("Falling back to author-based recommendations...")
                        recommendations = get_author_based_recommendations(selected_book, books, n=num_recs)
                        recommendation_title = "👤 More Books by Same Author (Fallback)"
                
                elif author_recs:
                    recommendations = get_author_based_recommendations(selected_book, books, n=num_recs)
                    recommendation_title = "👤 More Books by Same Author"
                
                elif popular_recs:
                    recommendations = get_popular_recommendations(books, ratings, n=num_recs)
                    recommendation_title = "⭐ Currently Popular Books"
                
                elif year_recs:
                    recommendations = get_year_based_recommendations(selected_book, books, n=num_recs)
                    recommendation_title = "📅 Books from the Same Era"
                
                elif similar_title_recs:
                    recommendations = get_title_similarity_recommendations(selected_book, books, n=num_recs)
                    recommendation_title = "📝 Books with Similar Titles"
                
                elif random_recs:
                    recommendations = get_random_recommendations(books, n=num_recs)
                    recommendation_title = "🎲 Random Book Discovery"
                
                # Display recommendations
                if not recommendations.empty:
                    st.subheader(recommendation_title)
                    
                    for i, book in recommendations.iterrows():
                        # Handle different dataframe structures
                        book_title = book.get('Book-Title', '')
                        book_author = book.get('Book-Author', '')
                        image_url = book.get('Image-URL-M', '')
                        year = book.get('Year-Of-Publication', '')
                        isbn = book.get('ISBN', '')
                        similarity_score = book.get('Similarity-Score', None)
                          # For popular recommendations, we might have different column names
                        if 'avg_rating' in book:
                            rating = book.get('avg_rating', None)
                            num_ratings = book.get('num_ratings', None)
                        else:
                            rating = None
                            num_ratings = None
                        
                        display_enhanced_book_card(
                            book_title,
                            book_author,
                            image_url,
                            rating=rating,
                            num_ratings=num_ratings,
                            year=year,
                            similarity_score=similarity_score,
                            isbn=isbn,
                            context="recommendations"
                        )
                        
                elif any([smart_recs, author_recs, popular_recs, year_recs, similar_title_recs, random_recs]):
                    st.warning(f"No recommendations found using the selected method. Try a different recommendation type!")
                
            else:
                st.error("Book details not found")
        else:
            st.info("👆 Search for a book above to get started with recommendations!")

if __name__ == "__main__":
    main()
