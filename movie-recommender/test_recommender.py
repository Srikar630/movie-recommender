import pytest
from recommender import load_data, build_similarity, recommend

df = load_data()
sim = build_similarity(df)


def test_returns_n_results_excluding_self():
    recs = recommend("Inception", df, sim, n=5)
    assert len(recs) == 5
    assert all(title != "Inception" for title, _ in recs)


def test_case_insensitive():
    assert recommend("inception", df, sim, n=3)


def test_unknown_movie():
    with pytest.raises(ValueError):
        recommend("Not A Real Movie", df, sim)
