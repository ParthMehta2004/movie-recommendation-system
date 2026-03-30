import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎥",
    layout="centered"
)


@st.cache_data
def load_data():
    """
    Load movies and ratings CSV files from the repo root.
    Both files must be present alongside app.py.
    """
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings


@st.cache_data
def build_similarity_matrix(movies, ratings):
    """
    Merge datasets, build a user-item matrix, and compute
    item-based cosine similarity between movies.
    """
    
    data = pd.merge(ratings, movies, on="movieId")

    
    user_item = data.pivot_table(
        index="userId",
        columns="title",
        values="rating"
    ).fillna(0)

    
    movie_matrix = user_item.T
    similarity = cosine_similarity(movie_matrix)

    
    sim_df = pd.DataFrame(
        similarity,
        index=movie_matrix.index,
        columns=movie_matrix.index
    )
    return sim_df


def recommend_movies(movie_title, sim_df, top_n=5):
    """
    Return a list of top_n recommended movie titles similar
    to the given movie. Returns None if the movie is not found.
    """
    if movie_title not in sim_df.index:
        return None

    scores = sim_df[movie_title].sort_values(ascending=False)
    scores = scores.drop(labels=[movie_title])   
    return scores.head(top_n).index.tolist()



def main():
    st.title("🎥 Movie Recommendation System")
    st.markdown(
        "Enter a movie name and click **Recommend** to discover "
        "similar movies using collaborative filtering."
    )
    st.markdown("---")

    
    try:
        movies, ratings = load_data()
    except FileNotFoundError:
        st.error(
            "Dataset files not found. Please make sure `movies.csv` and "
            "`ratings.csv` are present in the same directory as `app.py`."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    
    try:
        sim_df = build_similarity_matrix(movies, ratings)
    except Exception as e:
        st.error(f"Error building recommendation engine: {e}")
        st.stop()

    
    movie_input = st.text_input(
        "🔍 Enter a movie name",
        placeholder="e.g. Toy Story (1995)"
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=1, max_value=20, value=5
    )

    
    if st.button("✨ Recommend"):
        if not movie_input.strip():
            st.warning("Please enter a movie name first.")
            return

        
        matches = [
            title for title in sim_df.index
            if movie_input.strip().lower() in title.lower()
        ]

        if not matches:
            st.error(
                f"Movie **'{movie_input}'** not found in the dataset. "
                "Please check the title or try another movie."
            )
            return

        selected = matches[0]   
        recommendations = recommend_movies(selected, sim_df, top_n=top_n)

        if not recommendations:
            st.info("No similar movies found. Try a different title.")
            return

        st.success(f"Top {top_n} movies similar to **{selected}**:")
        for i, movie in enumerate(recommendations, start=1):
            st.markdown(f"**{i}.** {movie}")


if __name__ == "__main__":
    main()
