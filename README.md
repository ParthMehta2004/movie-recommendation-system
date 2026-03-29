# Movie Recommendation System

A Movie Recommendation System built with **Python** and **Streamlit** using item-based collaborative filtering and cosine similarity on the MovieLens small dataset.

---

## Features

- Enter any movie name to get similar movie recommendations
- Item-based collaborative filtering using cosine similarity
- Partial/case-insensitive movie name matching
- Adjustable number of recommendations (1–20)
- Clean error handling if movie is not found
- Deployed on Streamlit Community Cloud

---

## Dataset

This app uses the [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/).

Place these two files in the **root** of the project (same folder as `app.py`):

- `movies.csv`
- `ratings.csv`

---

## File Structure

```
movie-recommendation-system/
|-- app.py             # Main Streamlit application
|-- requirements.txt   # Python dependencies
|-- README.md          # Project documentation
|-- movies.csv         # MovieLens movies dataset
|-- ratings.csv        # MovieLens ratings dataset
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/ParthMehta2004/movie-recommendation-system.git
cd movie-recommendation-system
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Add dataset files**

Download `movies.csv` and `ratings.csv` from [MovieLens](https://grouplens.org/datasets/movielens/latest/) and place them in the root folder.

---

## Run Locally

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Deploy on Streamlit Community Cloud

1. Push this project to a **public GitHub repository**
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy**

Streamlit Cloud will automatically install packages from `requirements.txt` and run:

```bash
streamlit run app.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web app UI |
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | Cosine similarity computation |

---

## How It Works

1. Loads `movies.csv` and `ratings.csv`
2. Merges datasets and builds a **user-item rating matrix**
3. Transposes the matrix to get a **movie-user matrix**
4. Computes **cosine similarity** between all pairs of movies
5. For a given input movie, returns the **top N most similar movies**

---

## Author

**Parth Mehta** — [GitHub](https://github.com/ParthMehta2004)
