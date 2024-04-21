import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pickle import load, dump 
from collections import defaultdict

def predict(user_rated_movies):
  movie_train= pd.read_csv(r"C:\Users\skander\Desktop\pfa_env\the_app\model_data\x")
  movie_train_arr=np.genfromtxt(r'C:\Users\skander\Desktop\pfa_env\the_app\model_data\x', delimiter=',')
  num_user_features = 20
  num_item_features = 22

  num_outputs = 38
  tf.random.set_seed(1)
  user_NN = tf.keras.models.Sequential([

          keras.layers.Dense(512, activation='relu'),
          keras.layers.Dense(256, activation='relu'),
          keras.layers.Dense(num_outputs, activation=None)     ])

  # create the user input and point to the base network
  input_user = tf.keras.layers.Input(shape=(num_user_features))
  vu = user_NN(input_user)
  vu = tf.linalg.l2_normalize(vu, axis=1)


  item_NN = tf.keras.models.Sequential([

          keras.layers.Dense(512, activation='relu'),
          keras.layers.Dense(256, activation='relu'),
          keras.layers.Dense(num_outputs, activation=None)     ])

  # create the item input and point to the base network
  input_item = tf.keras.layers.Input(shape=(num_item_features))
  vm = item_NN(input_item)
  vm = tf.linalg.l2_normalize(vm, axis=1)

  # compute the dot product of the two vectors vu and vm
  output = tf.keras.layers.Dot(axes=1)([vu, vm])

  # specify the inputs and output of the model
  model = tf.keras.Model([input_user, input_item], output)

  # model.summary()

  checkpoint_path = r"C:\Users\skander\Desktop\pfa_env\the_app\model_data\param\cp.ckpt"

  model.load_weights(checkpoint_path)


  df3=pd.read_csv(r"C:\Users\skander\Desktop\pfa_env\the_app\model_data\df3")


  df3.drop('Unnamed: 0', axis=1, inplace=True)


# property
# user_rated_movies={"Dumb & Dumber (Dumb and Dumber) (1994)": 5,
#                    "Home Alone (1990)": 5,
#                    "Titanic (1997)":0.5,
#                    "Call Me by Your Name (2017)": 1,
#                    "Hangover, The (2009)": 4.5}

  user_dataframe=pd.DataFrame()
  for key,value in user_rated_movies.items() :
     serie=df3[df3['title'] == key].iloc[0]
     serie['rating']=value
     serie['userId']=5000
  #   print(serie)
     user_dataframe=pd.concat([user_dataframe,serie], axis=1)
  user_dataframe=user_dataframe.T.reset_index(drop=True)
  new_user_ratings = user_dataframe.groupby('userId')["rating"].agg(["mean", "count"]).rename(columns={"mean": "avg_rtg", "count": "rtg_cnt"})

  # Calculate the average rating for each genre for each user
  genre_ratings = user_dataframe.groupby("userId").apply(lambda x: x.iloc[:, 6:].multiply(x["rating"], axis=0).sum() / x.iloc[:, 6:].sum()).fillna(0)

  # Combine the mean ratings and genre ratings into a single dataframe
  user_train = pd.concat([new_user_ratings, genre_ratings], axis=1).fillna(0)

  user_train.reset_index(inplace=True)

  user_vec=user_train.to_numpy(dtype=float)

  user_train


  num_movies = 9257
  user_vecs = np.tile(user_vec, (num_movies, 1))



  scalerItem = load(open(r'C:\Users\skander\Desktop\pfa_env\the_app\model_data\scalerItem.pkl', 'rb'))
  scalerUser = load(open(r'C:\Users\skander\Desktop\pfa_env\the_app\model_data\scalerUser.pkl', 'rb'))
  scalerTarget = load(open(r'C:\Users\skander\Desktop\pfa_env\the_app\model_data\scalerTarget.pkl', 'rb'))

  # generate a 2 dim array of replicated user vector to match the number movies in the data set.
  user_vecs = np.tile(user_vec, (num_movies, 1))

  # scale our user and item vectors
  movie_vecs = movie_train.drop_duplicates().apply(pd.to_numeric)
  movie_vecs_arr=np.array(movie_vecs)
  sitem_vecs = scalerItem.transform(movie_vecs_arr)



  # user_vecs = user_train.drop_duplicates().apply(pd.to_numeric)#, errors='coerce')
  # user_vecs_arr=np.array(user_vecs)
  suser_vecs = scalerUser.transform(user_vecs)


  # make a prediction
  y_p = model.predict([suser_vecs[:, 3:], sitem_vecs[:, 1:]])
  print(type(y_p))
  # unscale y prediction
  y_pu = scalerTarget.inverse_transform(y_p)

  # sort the results, highest prediction first
  sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
  sorted_ypu   = y_pu[sorted_index]
  sorted_items = movie_vecs_arr[sorted_index]  #using unscaled vectors for display

  df_movies=pd.read_csv(r'C:\Users\skander\Desktop\pfa_env\the_app\model_data\df_movies')

  df_movies.drop('Unnamed: 0', axis=1, inplace=True)



  movie_dict = defaultdict(dict)
  for idx, row in df_movies.iterrows():
      movie_id = int(row['movieId'])
      movie_dict[movie_id]['title'] = row['title']
      movie_dict[movie_id]['genres'] = row['genres']

  maxcount=10
  count=0

  disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

  for i in range(0, sorted_ypu.shape[0]):
          if count == maxcount:
              break
          count += 1
          movie_id = sorted_items[i, 0].astype(int)
          disp.append([np.around(sorted_ypu[i, 0], 1), sorted_items[i, 0].astype(int), np.around(sorted_items[i, 2].astype(float), 1),
                       movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])
  disp=disp[1:6]
  dic = {}
  for sublist in disp:
    dic[sublist[3]] = sublist[0]
  return dic



user_rated_movies={"Dumb & Dumber (Dumb and Dumber) (1994)": 5,
                   "Home Alone (1990)": 5,
                   "Titanic (1997)":0.5,
                   "Call Me by Your Name (2017)": 1,
                   "Hangover, The (2009)": 4.5}


print(type(predict(user_rated_movies)))