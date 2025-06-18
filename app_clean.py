import streamlit as st
import numpy as np
import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="📚 Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #1E3A8A;
        }
        .stButton>button {
            background-color: #1E3A8A;
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2563EB;
            transform: translateY(-2px);
        }
        .book-card {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .book-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        .book-title {
            font-weight: bold;
            font-size: 18px;
            color: #1E3A8A;
            margin-bottom: 8px;
        }
        .book-author {
            font-style: italic;
            color: #6B7280;
            margin-bottom: 10px;
        }
        .rating {
            color: #FCD34D;
            font-size: 16px;
        }
        .recommendation-score {
            background: linear-gradient(90deg, #10B981, #059669);
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Functions to load and process data
@st.cache_data
def load_data():
    """Load the book data"""
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        
        books = pd.read_csv(os.path.join(data_dir, 'Books.csv'), low_memory=False, dtype={'Year-Of-Publication': 'str'})
        ratings = pd.read_csv(os.path.join(data_dir, 'Ratings.csv'))
        users = pd.read_csv(os.path.join(data_dir, 'Users.csv'))
        
        # Clean data
        books = books.dropna(subset=['Book-Title', 'Book-Author'])
        ratings = ratings.dropna()
        
        return books, ratings, users
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_preprocessed_data():
    """Load preprocessed data if available"""
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        with open(os.path.join(model_dir, 'top_books.pkl'), 'rb') as f:
            top_books = pickle.load(f)
        
        with open(os.path.join(model_dir, 'pt.pkl'), 'rb') as f:
            pt = pickle.load(f)
        
        return top_books, pt
    except Exception as e:
        st.sidebar.warning("Preprocessed data not found. Using live calculations.")
        return None, None

@st.cache_data
def prepare_popularity_based_data(books, ratings, top_books=None):
    """Get popular books data"""
    if top_books is not None:
        return top_books
        
    # Calculate from scratch
    ratings_with_name = ratings.merge(books, on='ISBN')
    
    # Count ratings per book
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
    
    # Calculate average rating per book
    avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
    
    # Create popularity dataframe
    popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    
    # Filter and sort
    popular_books = popularity_df[popularity_df['num_ratings'] >= 100].sort_values('avg_rating', ascending=False)
    
    # Merge with books
    top_books = popular_books.merge(books, on='Book-Title').drop_duplicates('Book-Title')
    selected_columns = ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating', 'Year-Of-Publication']
    
    return top_books[selected_columns].head(50)

def get_book_cover_api(isbn=None, title=None, author=None):
    """Fetch book cover from APIs"""
    # Try Open Library first
    if isbn:
        try:
            url = f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and len(response.content) > 1000:
                return url
        except:
            pass
    
    # Try Google Books API
    if title and author:
        try:
            query = f"{title} {author}".replace(' ', '+')
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    volume = data['items'][0]
                    if 'volumeInfo' in volume and 'imageLinks' in volume['volumeInfo']:
                        return volume['volumeInfo']['imageLinks'].get('thumbnail', '')
        except:
            pass
    
    # Fallback placeholder
    return "https://via.placeholder.com/150x200?text=No+Cover"

def display_book_card(book_title, book_author, image_url, rating=None, num_ratings=None, year=None, similarity_score=None, isbn=None):
    """Display a book card with enhanced styling"""
    with st.container():
        st.markdown('<div class="book-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Get better cover image
            try:
                better_image_url = get_book_cover_api(isbn, book_title, book_author)
                if better_image_url and better_image_url.startswith('http'):
                    st.image(better_image_url, width=120)
                else:
                    st.image("https://via.placeholder.com/120x180?text=No+Cover", width=120)
            except:
                st.image("https://via.placeholder.com/120x180?text=No+Cover", width=120)
        
        with col2:
            st.markdown(f'<p class="book-title">{book_title}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="book-author">by {book_author}</p>', unsafe_allow_html=True)
            
            if year:
                st.text(f"📅 Published: {year}")
                
            if rating:
                stars = "⭐" * min(int(rating), 5)
                st.markdown(f'<p class="rating">{stars} ({rating:.1f}/5)</p>', unsafe_allow_html=True)
            
            if num_ratings:
                st.text(f"📊 {num_ratings:,} ratings")
                
            if similarity_score:
                st.markdown(f'<span class="recommendation-score">Match: {similarity_score*100:.1f}%</span>', unsafe_allow_html=True)
            
            # Add to favorites button
            if st.button(f"❤️ Add to Favorites", key=f"fav_{book_title[:20]}"):
                if book_title not in st.session_state.favorites:
                    st.session_state.favorites.append(book_title)
                    st.success("Added to favorites!")
                else:
                    st.info("Already in favorites!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def get_collaborative_recommendations(selected_book, pt, books, n=5):
    """Get collaborative filtering recommendations"""
    if selected_book not in pt.index:
        return pd.DataFrame()
    
    # Get the book's ratings vector
    book_ratings = pt.loc[selected_book].values.reshape(1, -1)
    
    # Calculate cosine similarity with all books
    similarities = cosine_similarity(book_ratings, pt.values)[0]
    
    # Get most similar books
    similar_indices = similarities.argsort()[-n-1:-1][::-1]
    similar_books = []
    
    for idx in similar_indices:
        if similarities[idx] > 0:  # Only include books with positive similarity
            book_title = pt.index[idx]
            book_info = books[books['Book-Title'] == book_title].iloc[0] if len(books[books['Book-Title'] == book_title]) > 0 else None
            
            if book_info is not None:
                similar_books.append({
                    'Book-Title': book_title,
                    'Book-Author': book_info['Book-Author'],
                    'Image-URL-M': book_info['Image-URL-M'],
                    'Year-Of-Publication': book_info['Year-Of-Publication'],
                    'ISBN': book_info['ISBN'],
                    'Similarity-Score': similarities[idx]
                })
    
    return pd.DataFrame(similar_books)

def get_random_recommendations(books, n=5):
    """Get random book recommendations"""
    return books.sample(n=min(n, len(books)))

def get_popular_recommendations(top_books, n=5):
    """Get popular book recommendations"""
    return top_books.head(n)

def get_genre_author_recommendations(books, selected_book, n=5):
    """Get recommendations based on genre/author"""
    book_info = books[books['Book-Title'] == selected_book]
    if book_info.empty:
        return pd.DataFrame()
    
    author = book_info.iloc[0]['Book-Author']
    
    # Find books by same author
    same_author_books = books[books['Book-Author'] == author]
    
    # If not enough books by same author, get random books
    if len(same_author_books) < n:
        additional_books = books[books['Book-Author'] != author].sample(n=n-len(same_author_books))
        same_author_books = pd.concat([same_author_books, additional_books])
    
    return same_author_books.head(n)

# Main application
def main():
    # Sidebar
    st.sidebar.title("📚 Book Recommendation System")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio("📋 Navigation", [
        "🏠 Home", 
        "⭐ Popular Books", 
        "🎯 Book Recommendations",
        "❤️ My Favorites",
        "📊 Analytics"
    ])
    
    # Load data
    with st.spinner("Loading data..."):
        books, ratings, users = load_data()
        if books is None:
            st.error("Failed to load data. Please check your data files.")
            return
    
    st.sidebar.success(f"✅ Loaded {len(books):,} books, {len(ratings):,} ratings")
    
    # Home page
    if page == "🏠 Home":
        st.title("📚 Welcome to Book Recommendation System")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Discover Your Next Favorite Book!
            
            This intelligent recommendation system helps you find books you'll love using:
            
            - **🔥 Popular Books**: Highly rated books loved by thousands
            - **🎯 Smart Recommendations**: AI-powered suggestions based on your preferences
            - **🎲 Random Discovery**: Serendipitous book discoveries
            - **📚 Genre/Author Based**: Find books by your favorite authors or genres
            
            ### 📊 Dataset Statistics
            """)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📚 Books", f"{len(books):,}")
            with col_b:
                st.metric("👥 Users", f"{len(users):,}")
            with col_c:
                st.metric("⭐ Ratings", f"{len(ratings):,}")
        
        with col2:
            st.image("https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop", 
                    caption="Discover amazing books!")
    
    # Popular Books page
    elif page == "⭐ Popular Books":
        st.title("⭐ Most Popular Books")
        st.markdown("Books that are highly rated by many readers")
        
        # Load popular books
        top_books, _ = load_preprocessed_data()
        popular_books = prepare_popularity_based_data(books, ratings, top_books)
        
        # Filters
        col1, col2 = st.columns([1, 1])
        with col1:
            min_ratings = st.slider("Minimum number of ratings", 10, 500, 100)
        with col2:
            num_books = st.slider("Number of books to show", 5, 20, 10)
        
        # Filter books
        if 'num_ratings' in popular_books.columns:
            filtered_books = popular_books[popular_books['num_ratings'] >= min_ratings].head(num_books)
        else:
            filtered_books = popular_books.head(num_books)
        
        # Display books
        st.markdown("---")
        for idx, book in filtered_books.iterrows():
            display_book_card(
                book['Book-Title'],
                book['Book-Author'],
                book.get('Image-URL-M', ''),
                book.get('avg_rating'),
                book.get('num_ratings'),
                book.get('Year-Of-Publication')
            )
    
    # Book Recommendations page
    elif page == "🎯 Book Recommendations":
        st.title("🎯 Get Book Recommendations")
        st.markdown("Find books similar to ones you've enjoyed!")
        
        # Load collaborative filtering data
        _, pt = load_preprocessed_data()
        
        if pt is not None:
            # Book selection
            st.subheader("📖 Select a Book You Enjoyed")
            
            available_books = sorted(pt.index.tolist())
            
            # Search functionality
            search_term = st.text_input("🔍 Search for a book title:")
            
            if search_term:
                matched_books = [book for book in available_books if search_term.lower() in book.lower()]
                if matched_books:
                    selected_book = st.selectbox("Select from search results:", matched_books)
                else:
                    st.warning("No books found matching your search.")
                    selected_book = None
            else:
                selected_book = st.selectbox("Or choose from all books:", available_books)
            
            if selected_book:
                # Show selected book
                book_details = books[books['Book-Title'] == selected_book]
                if not book_details.empty:
                    st.subheader("📚 Your Selected Book")
                    book_info = book_details.iloc[0]
                    display_book_card(
                        book_info['Book-Title'],
                        book_info['Book-Author'],
                        book_info.get('Image-URL-M', ''),
                        year=book_info.get('Year-Of-Publication'),
                        isbn=book_info.get('ISBN')
                    )
                
                # Recommendation type selection
                st.subheader("🎯 Choose Recommendation Type")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    smart_recs = st.button("🧠 Smart Recommendations", use_container_width=True)
                with col2:
                    popular_recs = st.button("🔥 Popular Books", use_container_width=True)
                with col3:
                    genre_recs = st.button("📚 Genre/Author Based", use_container_width=True)
                with col4:
                    random_recs = st.button("🎲 Random Discovery", use_container_width=True)
                
                # Number of recommendations
                num_recs = st.slider("Number of recommendations:", 3, 10, 5)
                
                # Generate recommendations based on selection
                recommendations = pd.DataFrame()
                recommendation_title = ""
                
                if smart_recs:
                    with st.spinner("Generating smart recommendations..."):
                        recommendations = get_collaborative_recommendations(selected_book, pt, books, num_recs)
                        recommendation_title = "🧠 Smart Recommendations Based on Similar Users"
                
                elif popular_recs:
                    top_books, _ = load_preprocessed_data()
                    popular_books = prepare_popularity_based_data(books, ratings, top_books)
                    recommendations = get_popular_recommendations(popular_books, num_recs)
                    recommendation_title = "🔥 Most Popular Books"
                
                elif genre_recs:
                    recommendations = get_genre_author_recommendations(books, selected_book, num_recs)
                    recommendation_title = "📚 Books by Same Author or Similar Genre"
                
                elif random_recs:
                    recommendations = get_random_recommendations(books, num_recs)
                    recommendation_title = "🎲 Random Book Discovery"
                
                # Display recommendations
                if not recommendations.empty:
                    st.subheader(recommendation_title)
                    st.markdown("---")
                    
                    for idx, book in recommendations.iterrows():
                        display_book_card(
                            book['Book-Title'],
                            book['Book-Author'],
                            book.get('Image-URL-M', ''),
                            year=book.get('Year-Of-Publication'),
                            similarity_score=book.get('Similarity-Score'),
                            isbn=book.get('ISBN')
                        )
                else:
                    if smart_recs or popular_recs or genre_recs or random_recs:
                        st.warning("No recommendations found. Please try a different recommendation type.")
        
        else:
            st.warning("Collaborative filtering data not available. Please run preprocessing first.")
    
    # Favorites page
    elif page == "❤️ My Favorites":
        st.title("❤️ My Favorite Books")
        
        if st.session_state.favorites:
            st.success(f"You have {len(st.session_state.favorites)} favorite books!")
            
            for fav_book in st.session_state.favorites:
                book_info = books[books['Book-Title'] == fav_book]
                if not book_info.empty:
                    book = book_info.iloc[0]
                    display_book_card(
                        book['Book-Title'],
                        book['Book-Author'],
                        book.get('Image-URL-M', ''),
                        year=book.get('Year-Of-Publication'),
                        isbn=book.get('ISBN')
                    )
            
            if st.button("🗑️ Clear All Favorites"):
                st.session_state.favorites = []
                st.success("Favorites cleared!")
        else:
            st.info("No favorite books yet. Add some books to your favorites from the recommendations!")
    
    # Analytics page
    elif page == "📊 Analytics":
        st.title("📊 Reading Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Your Statistics")
            st.metric("❤️ Favorite Books", len(st.session_state.favorites))
            st.metric("⭐ Books Rated", len(st.session_state.user_ratings))
            
            if st.session_state.favorites:
                st.subheader("🔖 Your Favorite Books")
                for book in st.session_state.favorites:
                    st.write(f"• {book}")
        
        with col2:
            st.subheader("📊 Dataset Overview")
            
            # Show dataset statistics
            if not books.empty:
                # Top authors by number of books
                top_authors = books['Book-Author'].value_counts().head(10)
                st.subheader("📚 Most Prolific Authors")
                for author, count in top_authors.items():
                    st.write(f"• {author}: {count} books")

if __name__ == "__main__":
    main()
