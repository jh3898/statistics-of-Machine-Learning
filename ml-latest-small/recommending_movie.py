import os

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt
ratings = pd.read_csv('ratings.csv')
print(ratings.head())

movies = pd.read_csv('movies.csv')


#movies['category'] = \
#movies = pd.get_dummies(movies['genres'],prefix= 'genr')
#mov_columns = movies[['movieId', 'title']]
#movies_new = pd.merge([['dummy_genres'], mov_columns])
print(movies.head())

# Combining movie ratings & movie names
ratings = pd.merge(ratings[['userId', 'movieId', 'rating']], movies[['movieId','title']],how = 'left',left_on = 'movieId',right_on='movieId')
rp = ratings.pivot_table(columns= ['movieId'], index = ['userId'], values = 'rating')
rp = rp.fillna(0)
rp_mat = rp.as_matrix()

a = np.asarray([2,1, 0,2,0,1,1,1])
b = np.array([2,1,1,1,1,0,1,1])
print(1- cosine(a,b))
m ,n = rp.shape
# User similarity matrix
mat_users = np.zeros((m,m))

for i in range(m):
    for j in range(m):
        if i != j:
            mat_users[i][j] = (1- cosine(rp_mat[i,:], rp_mat[j,:]))
        else:
            mat_users[i][j] =0

pd_users = pd.DataFrame(mat_users, index= rp.index, columns= rp.index)

# Finding similar users
def topn_simusers(uid = 16, n = 5):
    




