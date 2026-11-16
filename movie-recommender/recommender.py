import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path="movies.csv"):
    df = pd.read_csv(path)
    df["soup"] = (df["genres"] + " " + df["genres"] + " " + df["keywords"]).str.lower()
    return df


def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["soup"])
    return cosine_similarity(matrix)


def recommend(title, df, sim, n=5):
    matches = df.index[df["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        raise ValueError(f"Movie not found: {title}")
    idx = matches[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    top = [(df.loc[i, "title"], round(float(s), 3)) for i, s in scores if i != idx][:n]
    return top


if __name__ == "__main__":
    import sys

    df = load_data()
    sim = build_similarity(df)
    query = " ".join(sys.argv[1:]) or "Inception"
    print(f"Because you liked {query}:")
    for name, score in recommend(query, df, sim):
        print(f"  {name}  (similarity {score})")
