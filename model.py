# #Import Libraries
# import numpy as np
# import pandas as pd
# import joblib
# import pickle
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
#
# # load data
# df = pd.read_csv("ohe_data_reduce_cat_class.csv")
#
# # Split data
# X= df.drop('price', axis=1)
# y= df['price']
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=51)
#
# # feature scaling
# sc = StandardScaler()
# sc.fit(X_train)
#
# # model_reg.fit(scaled_x_train.values, y_train[vp].values)
# # data_pred = model_reg.predict(scaled_x_test.values)
#
# X_train = sc.transform(X_train)
# X_test = sc.transform(X_test)
#
#
# ###### Load Model
#
# model = joblib.load('bangalore_house_price_prediction_model.pkl')
# # model=pickle.load(open("bangalore_house_price_prediction_model.pkl",'rb'))
#
#
#
# # it help to get predicted value of house  by providing features value
# def predict_house_price(bath, balcony, total_sqft_int, bhk, price_per_sqft, area_type, availability, location):
#
#   x =np.zeros(len(X.columns)) # create zero numpy array, len = 107 as input value for model
#
#   # adding feature's value accorind to their column index
#   x[0]=bath
#   x[1]=balcony
#   x[2]=total_sqft_int
#   x[3]=bhk
#
#   print("bath :"+bath+"balcony"+balcony+"total_sqft"+total_sqft_int+"BHK :"+bhk+"Price Per SQFT"+price_per_sqft+"Area Type"+area_type+"Availibilty"+availability+"Location"+location)
#
#   if availability=="Ready To Move":
#     x[8]=1
#
#   if 'area_type'+area_type in X.columns:
#     area_type_index = np.where(X.columns=="area_type"+area_type)[0][0]
#     x[area_type_index] =1
#
#   if (price_per_sqft=="" or  price_per_sqft== 0)  :
#      x[4]=np.mean(X['price_per_sqft'])
#   else:
#     print("hiii")
#     x[4]=price_per_sqft
#
#   if 'location_'+location in X.columns:
#     loc_index = np.where(X.columns=="location_"+location)[0][0]
#     x[loc_index]=1
#
#   # feature scaling
#   x = sc.transform([x])[0] # give 2d np array for feature scaling and get 1d scaled np array
#   return model.predict([x])[0] # return the predicted value by train XGBoost model
# Import Libraries
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv("ohe_data_reduce_cat_class.csv")

# Split data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

# feature scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

###### Load Model

# model = joblib.load('bangalore_house_price_prediction_model.pkl')
models = pickle.load(open("bangalore_house_price_prediction_model.pkl",'rb'))

# it help to get predicted value of house  by providing features value
def predict_house_price(bath, balcony, total_sqft_int, bhk, price_per_sqft, area_type, availability, location):
  x = np.zeros(len(X.columns))  # create zero numpy array, len = 107 as input value for model

  # adding feature's value accorind to their column index
  x[1] = bath
  x[2] = balcony
  x[3] = total_sqft_int
  x[4] = bhk

  if availability == "Ready To Move":
    x[9] = 1

  if 'area_type' + area_type in X.columns:
    area_type_index = np.where(X.columns == "area_type" + area_type)[0][0]
    x[area_type_index] = 1

  if 'location_' + location in X.columns:
    loc_index = np.where(X.columns == "location_" + location)[0][0]
    x[loc_index] = 1

  if price_per_sqft==0 or price_per_sqft=="" :
    x[5]=np.mean(X['price_per_sqft'])
  else:
    x[5]=price_per_sqft

  if area_type == None or area_type =='':
     x[6]=1

  if availability == None or availability == '':
    x[10] = 1
  # feature scaling
  x = sc.transform([x])[0]  # give 2d np array for feature scaling and get 1d scaled np array
  # prediction = pipe.predict(input)[0] * 1e5
  # str(np.round(prediction, 2))
  return str(np.round((models.predict([x])[0])*1e5,2))   # return the predicted value by train XGBoost model

