# 📚 Book Recommendation System

A sophisticated book recommendation system built with Streamlit that provides personalized book recommendations using collaborative filtering and popularity-based algorithms.

## 🌟 Features

- **Popularity-Based Recommendations**: Discover books that are highly rated by thousands of readers
- **Collaborative Filtering**: Get personalized recommendations based on books you already like
- **Interactive Search**: Search through thousands of books with real-time filtering
- **Beautiful UI**: Modern, responsive interface with book cards and ratings
- **Smart Caching**: Optimized performance with preprocessed data

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd Recommendation_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run preprocessing (first time only)**
   ```bash
   python preprocess.py
   ```
   This will create model files in the `models/` directory for faster loading.

4. **Start the application**
   ```bash
   streamlit run app.py
   ```
   Or double-click `run_app.bat` on Windows.

5. **Open your browser**
   The app will automatically open at `http://localhost:8501`

## 📊 Dataset

The system uses a comprehensive book dataset with:
- 📚 **271,360 books** with detailed metadata
- ⭐ **1,149,780 ratings** from real users  
- 👥 **278,858 users** who rated books

## 🔧 How It Works

### Popularity-Based Recommendations
- Analyzes books with high ratings and many reviews
- Filters books with minimum 250+ ratings for reliability
- Sorts by average rating to surface the best books

### Collaborative Filtering
- Uses cosine similarity to find books with similar rating patterns
- Filters active users (50+ ratings) and popular books (20+ ratings)
- Creates a user-item matrix and calculates book similarities

## 🎯 Usage

### Home Page
- Overview of the system and dataset statistics
- Navigation to different recommendation types

### Popular Books
- Browse highly-rated books
- Filter by minimum number of ratings
- View book details, ratings, and publication years

### Book Recommendations
- Search for a book you've enjoyed
- Get 5 personalized recommendations
- See similarity scores and detailed book information

## 📁 Project Structure

```
Recommendation_system/
├── app.py              # Main Streamlit application
├── preprocess.py       # Data preprocessing script
├── requirements.txt    # Python dependencies
├── run_app.bat        # Windows batch file to run app
├── data/              # Dataset files
│   ├── Books.csv      # Book metadata
│   ├── Ratings.csv    # User ratings
│   └── Users.csv      # User information
├── models/            # Preprocessed model files (created after first run)
│   ├── top_books.pkl  # Popular books data
│   ├── pt.pkl         # Pivot table for collaborative filtering
│   └── similarity_scores.pkl # Cosine similarity matrix
└── notebook/          # Jupyter notebook with analysis
```

## ⚡ Performance Tips

- The first run will be slower as it processes the data
- After preprocessing, the app loads much faster using cached model files
- To refresh recommendations, delete the `models/` folder and run `preprocess.py` again

## 🛠️ Customization

You can modify the recommendation parameters in `preprocess.py`:
- Change minimum rating thresholds
- Adjust user activity filters
- Modify similarity calculation methods

## 📝 Technical Details

- **Framework**: Streamlit for web interface
- **ML Libraries**: scikit-learn for similarity calculations
- **Data Processing**: pandas and numpy
- **Caching**: Streamlit's built-in caching for performance
- **UI**: Custom CSS for modern design

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

---

Enjoy discovering your next favorite book! 📖✨
