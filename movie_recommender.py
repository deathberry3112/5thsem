import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read in the movie data
movies = pd.read_csv('E:\jprograms\movies.csv')

# Create a list of movie descriptions
descriptions = movies['description'].tolist()

# Create a TfidfVectorizer to convert the descriptions into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(descriptions)

# Calculate the cosine similarity between the vectors
similarity = cosine_similarity(vectors)

# Function to recommend movies based on their descriptions
def recommend_movies(title, similarity=similarity):
  # Find the index of the movie in the dataframe
  idx = movies[movies['title'] == title].index[0]

  # Find the movie's similarity scores
  sim_scores = list(enumerate(similarity[idx]))

  # Sort the movies by their similarity scores
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  # Get the top 10 most similar movies
  sim_scores = sim_scores[1:11]

  # Get the movie IDs
  movie_ids = [i[0] for i in sim_scores]

  # Return the top 10 most similar movies
  return movies['title'].iloc[movie_ids]

# Test the recommend_movies function
print(recommend_movies('Inception'))
