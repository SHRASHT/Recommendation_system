import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.book-card {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        books = pd.read_csv(os.path.join(data_dir, 'Books.csv'), low_memory=False, nrows=1000)
        ratings = pd.read_csv(os.path.join(data_dir, 'Ratings.csv'), nrows=10000)
        users = pd.read_csv(os.path.join(data_dir, 'Users.csv'), nrows=1000)
        return books, ratings, users
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def display_book_card(title, author, year=None, rating=None):
    st.markdown(f"""
    <div class="book-card">
        <h4>{title}</h4>
        <p><em>by {author}</em></p>
        {f"<p>Year: {year}</p>" if year else ""}
        {f"<p>Rating: {rating:.1f}/5</p>" if rating else ""}
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("📚 Book Recommendation System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home", "Popular Books", "Test"])
    
    if page == "Home":
        st.write("Welcome to the Book Recommendation System!")
        
        # Load data
        books, ratings, users = load_data()
        
        if books is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Books", f"{len(books):,}")
            col2.metric("Ratings", f"{len(ratings):,}")
            col3.metric("Users", f"{len(users):,}")
            
            st.subheader("Sample Books")
            for _, book in books.head(5).iterrows():
                display_book_card(
                    book['Book-Title'], 
                    book['Book-Author'],
                    book.get('Year-Of-Publication')
                )
        else:
            st.error("Could not load data")
            
    elif page == "Popular Books":
        st.write("Popular Books Page")
        books, ratings, users = load_data()
        
        if books is not None and ratings is not None:
            # Simple popularity calculation
            ratings_with_name = ratings.merge(books, on='ISBN')
            popularity = ratings_with_name.groupby('Book-Title').agg({
                'Book-Rating': ['count', 'mean'],
                'Book-Author': 'first'
            }).reset_index()
            
            popularity.columns = ['Book-Title', 'num_ratings', 'avg_rating', 'Book-Author']
            popular_books = popularity[popularity['num_ratings'] >= 10].sort_values('avg_rating', ascending=False).head(10)
            
            st.subheader("Most Popular Books")
            for _, book in popular_books.iterrows():
                display_book_card(
                    book['Book-Title'],
                    book['Book-Author'],
                    rating=book['avg_rating']
                )
    
    elif page == "Test":
        st.write("This is a test page")
        st.success("App is working!")

if __name__ == "__main__":
    main()
