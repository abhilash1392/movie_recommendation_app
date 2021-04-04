# Importing the libraries
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

filepath = 'titles.csv'



def recommended_movie(movie_title):
    df = pd.read_csv(filepath)
    with open('matrix.npy','rb') as f:
        similarity_matrix=np.load(f)
    if movie_title not in list(df['title']):
        return []
    else:
        movie_index = df[df.title == movie_title].index
        similarity_score = similarity_matrix[movie_index]
        movie_frame = pd.DataFrame(similarity_score[0],columns=['cosine_similarity'])
        movie_frame=movie_frame.reset_index()
        movie_frame = movie_frame.sort_values(by='cosine_similarity',ascending=False)
        top10 = 10 
        top10_index = list(movie_frame['index'])[:10]
        recommended_movies = []
        for i in top10_index:
            temp = df['title'][i]
            recommended_movies.append(temp)
        return recommended_movies[1:]



