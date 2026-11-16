# 🎬 Movie Recommender

A content-based movie recommendation system. It converts each movie's genres and keywords into TF-IDF vectors and recommends the most similar titles using cosine similarity.

## Tech Stack
Python, pandas, scikit-learn, Streamlit

## Dataset
A small sample (`movies.csv`, 42 movies) is included so the project runs out of the box. To scale up, replace it with a larger dataset (e.g. TMDB 5000 Movies on Kaggle) keeping the columns `title, genres, keywords`.

## How to Run
```bash
pip install -r requirements.txt

# command line
python recommender.py Inception

# web app
streamlit run app.py
```

## Run Tests
```bash
pip install pytest
pytest
```

## Screenshot
_Add a screenshot here._

## Live Demo
_Add Streamlit Cloud link here._
