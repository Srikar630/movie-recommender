import streamlit as st
from recommender import load_data, build_similarity, recommend


@st.cache_data
def prepare():
    df = load_data()
    return df, build_similarity(df)


df, sim = prepare()

st.title("🎬 Movie Recommender")
st.caption("Content-based filtering using TF-IDF and cosine similarity")

choice = st.selectbox("Pick a movie you like:", sorted(df["title"]))
n = st.slider("How many recommendations?", 3, 10, 5)

if st.button("Recommend"):
    for name, score in recommend(choice, df, sim, n):
        st.write(f"**{name}**  ·  similarity {score:.2f}")
