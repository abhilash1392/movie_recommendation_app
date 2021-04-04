# Importing the libraries
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

filepath = 'titles.csv'

def get_data(path):
    df = pd.read_csv(path)
    for i in df.columns:
        df[i] = df[i].fillna("")
        df[i] = df[i].astype(str)

    df = df[['title','director','cast','country','listed_in','description']]
    for i in ['director','cast','country','description','listed_in']:
        df[i] = df[i].apply(lambda x:x.lower())
    
    df['director'] = df['director'].apply(lambda x: x.replace(','," "))
    df['cast'] = df['cast'].apply(lambda x: x.replace(','," "))
    df['listed_in'] = df['listed_in'].apply(lambda x: x.replace(','," "))
    df['total'] =df['title']+" "+df['listed_in']+" "+ df['director']+" "+df['cast']+" "+df['country']+" "+df['description']
    tfv = CountVectorizer()
    text_features = tfv.fit_transform(df['total'])
    return cosine_similarity(text_features)


matrix = get_data(filepath)
with open('matrix.npy','wb') as f:
    np.save(f,matrix)


