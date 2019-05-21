
# coding: utf-8

# # **Stock Price Prediction using Neural Network, LSTM , CNN models**

# # Project: Stock Price Prediction using Neural Network | Anshul Jain**

#     
# The data has seven columns as follows: 
#  
# Date, Open, High, Low, Close, Adj_Close, Volume 
#  
# Note :date and adj_close columns has been removed first since you don’t need them.

# In[1]:


import pandas as pd 
import math
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras as kp
import keras
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import Dense, Dropout , Flatten
import sklearn as sk
from keras.models import Sequential
from keras.layers.core import  Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM
from keras.layers import Dropout, Activation, Flatten  
from keras.layers import Convolution2D, MaxPooling2D 
from keras.optimizers import Adam
import collections
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import  Embedding

from keras.datasets import imdb

import numpy as np


# **Importing Libraries**

# **Importing the Data :**

# In[2]:



#Importing Files to Colab
from google.colab import files
files.upload()


# In[ ]:


df = pd.read_csv("Stock_Price_MAX.csv")


# **Data Preprocessing**

# In[4]:


df.shape


# In[5]:


df.head()


# **Dropping Date and Adj_Close**

# In[ ]:


df =df.drop(columns=['Date','Adj_Close'])


# In[7]:


df.shape


# In[8]:


df.count()


# **Drop duplicate rows We need to drop duplicate rows as that ,may bias the model when training**

# In[ ]:


df.drop_duplicates(keep = 'first', inplace = True)


# In[10]:


df.count()


# **Checking for Outliers**

# In[11]:


figs, axess = plt.subplots(2,3)
figs.set_size_inches(18.5, 10.5)
sns.boxplot(x=df['Open'],ax=axess[0,0])
sns.boxplot(x=df['High'],ax=axess[0,1])
sns.boxplot(x=df['Low'],ax=axess[1,0])
sns.boxplot(x=df['Close'],ax=axess[1,1])
sns.boxplot(x=df['Volume'],ax=axess[0,2])
sns.distplot(df['Close'].dropna(),kde=True,ax=axess[1,2])


# In[12]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter( df['Volume'],df['Close'])
ax.set_xlabel('Volume Stock Price')
ax.set_ylabel('Closing Stock price')
plt.show()


# **Removing outliers with z-score > 3**

# In[13]:


df.count()


# In[14]:


df.count()


# **Dropping NaN values if any **

# In[ ]:


df.dropna(inplace= True)


# In[16]:


df.count()


# **Dropping Duplicate Rows if any **

# In[ ]:


df.drop_duplicates(keep = 'first', inplace = True)


# In[18]:


df.count()


# In[19]:


df.head()


# **Splitting the dataset into Training and Tesing set**

# In[ ]:


dff = df.drop(columns=['Close'])
dfl = df['Close']


# In[21]:


dff.head(2)


# **Normalising the dataset using MinMax**

# In[ ]:


# Standardizing the dataset 
x = df.values #returns a numpy array
sc = MinMaxScaler(feature_range =(0,1))
dfc = sc.fit_transform(dff)
dff = pd.DataFrame(dfc,columns=dff.columns)


# In[23]:


dfl.head(2)


# In[24]:


dff.head(2)


# **Task 1:  Use the daily [Open, High, Low, Volume] to predict [Close] on that day using a fullyconnected neural network.  Use the first 70% of the records for training and the remaining 30% of the records for test. Report the RMSE of the model.  Show the “regression lift chart” of your test data. ** 

# **Lift Chart defination**

# In[ ]:


# Regression chart.
def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
        a = plt.plot(t['y'].tolist(),label='expected',color='black', lw=2)
    b = plt.plot(t['pred'].tolist(),label='prediction',color='darkorange')
    plt.ylabel('output')
    plt.legend()
    plt.show()


# In[ ]:


def liftchart(pred3 , y_true):
    print("**Lift Chart**")
    pred = pd.DataFrame(data=pred3)
    pred.columns=['pred']
    y_t = pd.DataFrame(data=y_true)
    y_t.columns=['actual']
    dfi = pd.merge(left=pred, left_index=True, right=y_t, right_index=True,how='inner')
    pred_ranks = pd.qcut(dfi['pred'].rank(method='first'), 100, labels=False,duplicates='drop')
    actual_ranks = pd.qcut(dfi['actual'].rank(method='first'), 100, labels=False,duplicates='drop')
    pred_percentiles = dfi.groupby(pred_ranks).mean()
    actual_percentiles = dfi.groupby(actual_ranks).mean()
    plt.plot(np.arange(.01, 1.01, .01), np.array(pred_percentiles['pred']),color='darkorange', lw=2, label='Prediction')
    plt.title('Lift Chart')
    plt.plot(np.arange(.01, 1.01, .01), np.array(pred_percentiles['actual']),
             color='navy', lw=2, linestyle='--', label='Actual')
    plt.ylabel('Target Percentile')
    plt.xlabel('Population Percentile')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 110.05])
    plt.legend(loc="best")


# In[ ]:


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column. 
    target_type = df[target].dtypes
    target_type = target_type[0] if isinstance(target_type, collections.Sequence) else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df[result].values.astype(np.float32), df[target].values.astype(np.float32)


# In[ ]:


# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(dff, dfl, test_size=0.30, random_state=42)


# In[29]:


# Define ModelCheckpoint outside the loop
checkpoint1 =  ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
for i in range(2):
########################################################################################################################
  model1 = Sequential()
  model1.add(Dense(20, input_dim=dff.shape[1], activation='relu'))
  model1.add(Dense(10, activation='relu'))
  #model1.add(Dense(10, activation='relu'))
  model1.add(Dropout(0.2))
  model1.add(Dense(15, 
         kernel_regularizer=regularizers.l1(0.01),
         activity_regularizer=regularizers.l2(0.01), activation='relu'))
  model1.add(Dense(10, activation='relu'))
  model1.add(Dense(1,kernel_initializer='normal', activation ='linear'))
  model1.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
  monitor1 = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='auto')
  checkpointer1 = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # save best model
  callbacks_list = [monitor1,checkpoint1]
  model1.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor1,checkpointer1],verbose=2,epochs=500)


# In[30]:


print('Training finished...Loading the best model')  
model1.load_weights('best_weights.hdf5') # load weights from best model
# Measure accuracy
pred3 = model1.predict_proba(x_test)
score = sk.metrics.r2_score(y_test, pred3)
print("Final R2 Score: {}".format(score))
print("RMSE:",math.sqrt(mean_squared_error(y_test, pred3)) )
print("\n")
#liftchart(y_true, pred3)

a = np.array(y_test).reshape(y_test.shape[0],1).flatten()
chart_regression(pred3.flatten() ,a,sort=True )


# In[177]:


liftchart(y_test.sort_values(), pred3)


# **Task 2:  Predict [Close] of a day based on the last 7 days’ data [Open, High, Low, Volume, Close] using a LSTM model.  In other words, we want to predict the price in the green cell using all the numbers in the red cell. Use the first 70% of the available records for training and the remaining 30% of the available records for test. Report the RMSE of the model. Show the “regression lift chart” of your test data.**
# ![image.png](attachment:image.png)

# In[ ]:


def create_dataset(dataset,ydata,look_back):
    SEQUENCE_SIZE = look_back
    dataX, dataY = [], []
    for i in range(len(dataset)-SEQUENCE_SIZE-1):
        window = dataset[i:(i+SEQUENCE_SIZE)]
        after_window = ydata[i+SEQUENCE_SIZE+1]
        window = [[dataX] for dataX in window]
        dataX.append(window)
        dataY.append(after_window)
    return np.array(dataX), np.array(dataY)


# **create new data set with new "Close" column 

# In[ ]:


df['CloseY'] = df['Close']


# In[34]:


df.head(2)


# In[ ]:


dff = df.drop(columns = ['CloseY'] )
dfl = df['CloseY'].copy()
dfl.columns = ['CloseY']


# **Splitting the Data**

# In[ ]:


# split into train and test sets
train_size = int(len(dff) * 0.70)
test_size = len(dff) - train_size
train = dff.iloc[0:train_size,:]
test = dff.iloc[train_size:len(dff),:]
trainy = dfl.iloc[0:train_size]
testy = dfl.iloc[train_size:len(dfl)]


# **Normalize the input**

# In[ ]:


# Standardizing the dataset 
x = df.values #returns a numpy array
sc = MinMaxScaler(feature_range =(0,1))
dfc = sc.fit_transform(train)
train = pd.DataFrame(dfc,columns=train.columns)
dft = sc.fit_transform(test)
test = pd.DataFrame(dft,columns=test.columns)




# Normalising with Z score gives better result than min max scalar

# In[ ]:


from scipy.stats import zscore  #method 1 
train = train.apply(zscore)
test = test.apply(zscore)


# In[39]:


print("Train Data Feature shape : ", train.shape)
print("Test Data Feature shape :", test.shape)
print("Train Y ", trainy.shape)
print("Test Y , " ,testy.shape)


# In[ ]:


# split into train and test sets
look_back = 7
trainX, trainY = create_dataset(train.values,trainy,look_back)
testX, testY = create_dataset(test.values,testy.values,look_back)


# In[41]:


testy.head(11)


# In[42]:


testY


# In[43]:


print("Train Data Feature shape : ", trainX.shape)
print("Test Data Feature shape :", testX.shape)
print("Train Y ", trainY.shape)
print("Test Y , " ,testY.shape)


# In[ ]:


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 5))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 5))


# In[45]:


testX.shape


#  dropout=0.1, recurrent_dropout=0.1, 

# In[46]:


SEQUENCE_SIZE = 7


# Define ModelCheckpoint outside the loop
checkpointlstm =  ModelCheckpoint(filepath="best_weightslstm.hdf5", verbose=10, save_best_only=True,mode='max') # save best model

for i in range(2):
  print('Build model...')
  model2 = Sequential()
  model2.add(LSTM(100,input_shape=(SEQUENCE_SIZE, 5),return_sequences = True ))
  model2.add(Dropout(0.1))
  model2.add(Dense(32))
  model2.add(LSTM(100,return_sequences = False))
  model2.add(Dense(1))
  model2.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
  monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=10, mode='auto')
  print('Train...')
  callbacks_list = [monitor,checkpointlstm]
  model2.fit(trainX,trainY,validation_data=(testX,testY),callbacks= callbacks_list,verbose=2, epochs=100)  


# In[47]:


print('Training finished...Loading the best model')  
print()
model2.load_weights("best_weightslstm.hdf5") # load weights from best model
testPredict = model2.predict(testX,batch_size=7)
score = sk.metrics.r2_score(testY, testPredict)
print("Final R2 Score: {}".format(score))

print("RMSE:",math.sqrt(mean_squared_error(testY, testPredict)) )
print("\n")
#liftchart(testY, testPredict)
a = np.array(testY).reshape(testY.shape[0],1).flatten()
chart_regression(testPredict.flatten() ,testY,sort=True )


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

# ===================================================================================================================================================================================

# **CNN Model**

# **3 : Predict [Close] of a day based on the last 7 days’ data [Open, High, Low, Volume, Close] using a CNN model. In other words, we want to predict the price in the green cell using all the numbers in the red cell. Use the first 70% of the available records for training and the remaining 30% of the available records for test. Report the RMSE of the model. Show the “regression lift chart” of your test data.**

# In[48]:


dff.head()


# In[49]:


dfl.head()


# In[ ]:


# split into train and test sets at 70:30 ratio
train_size = int(len(dff) * 0.70)
test_size = len(dff) - train_size
xtrain = dff.iloc[0:train_size,:]
xtest = dff.iloc[train_size:len(dff),:]
ytrain = dfl.iloc[0:train_size]
ytest = dfl.iloc[train_size:len(dfl)]


# In[51]:


xtrain.head(2)


# In[52]:


xtest.head(2)


# In[53]:


ytrain.head(2)


# In[54]:


ytest.head(2)


# In[ ]:


# create the feature dataset to be fed into the model
look_back = 7
xtrain, ytrain = create_dataset(train.values,trainy,look_back)
xtest, ytest = create_dataset(test.values,testy.values,look_back)


# In[56]:


xtrain.shape


# In[ ]:


def cnnmodel( xtrain, ytrain,xtest,ytest,shapea, shapeb, shapec):
  num_classes = 2
  # Define ModelCheckpoint outside the loop
  checkpoint3 = ModelCheckpoint(filepath="best_weightscnn.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
  for i in range(3):
  ########################################################################################################################
     model = Sequential()
     model.add(Conv2D(32, kernel_size=(1, 1), padding='same',input_shape=(shapea, shapeb, shapec)))
     model.add(Activation('relu'))
     model.add(MaxPooling2D(pool_size=(1, 1)))
     model.add(Dropout(0.25))
     model.add(Conv2D(64, (1, 1), padding='same'))
     model.add(Activation('relu'))
     model.add(MaxPooling2D(pool_size=(1, 1)))
     model.add(Dropout(0.25))
     model.add(Flatten())
     model.add(Dense(512))
     model.add(Activation('relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1, activation="linear"))

     monitor3 = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=2, mode='auto')  
    

    # show not only log loss but also accuracy for each epoch using metrics=['accuracy']
     model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
    # The test set is checked during training to monitor progress for early stopping but is never used for gradient descent (model training)

     callbacks_list = [monitor3,checkpoint3]
     model.fit(xtrain, ytrain, validation_data=(xtest,ytest), callbacks=callbacks_list, verbose=2,batch_size=32, epochs=100)
  print('Training finished...Loading the best model')  
  print()
  model.load_weights("best_weightscnn.hdf5") # load weights from best model
  testPred = model.predict(xtest,batch_size=7)
  return testPred


# In[69]:


model = cnnmodel(xtrain, ytrain,xtest,ytest,7,1,5)


# In[72]:


testPred = model
score = sk.metrics.r2_score(ytest, testPred)
print("Final R2 Score: {}".format(score))

print("RMSE:",math.sqrt(mean_squared_error(ytest, testPred)) )
print("\n")
#liftchart(testY, testPredict)
a = np.array(ytest).reshape(ytest.shape[0],1).flatten()
chart_regression(testPred.flatten() ,ytest,sort=True )


# 
# ## **Additional Features **
#  
#  **In the project, you predict [Close] of a day based on the last 7 days’ data.   
#  Can you find the best N value (number of the days we should consider in the past) that yields the best model?  **
#  

# In[ ]:


#cnn Model
def modelp( xtrain, ytrain,xtest,ytest,shapea, shapeb, shapec):
  num_classes = 2
  # Define ModelCheckpoint outside the loop
  checkpointn = ModelCheckpoint(filepath="best_weightsp.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
  for i in range(3):
  ########################################################################################################################
     model = Sequential()
     model.add(Conv2D(32, kernel_size=(1, 1), padding='same',input_shape=(shapea, shapeb, shapec)))
     model.add(Activation('relu'))
     model.add(Flatten())
     model.add(Dense(1, activation="linear"))
     monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto' )
     callbacks_list = [monitor,checkpointn] 
     model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
     model.fit(xtrain,ytrain,validation_data=(xtest,ytest),callbacks=callbacks_list,verbose=0, epochs=50)
  
  model.load_weights("best_weightsp.hdf5")
  testPred = model.predict(xtest,batch_size=i)
  score = sk.metrics.r2_score(ytest, testPred)
  testPred = model.predict(xtest,batch_size=i)
  rmse = mean_squared_error(ytest, testPred)
  return score , rmse


# In[ ]:


# split into train and test sets at 70:30 ratio
train_size = int(len(dff) * 0.70)
test_size = len(dff) - train_size
nxtrain = dff.iloc[0:train_size,:]
nxtest = dff.iloc[train_size:len(dff),:]
nytrain = dfl.iloc[0:train_size]
nytest = dfl.iloc[train_size:len(dfl)]


# In[ ]:


start = 7
end = 20
c , w, h =0, 3, end;
result  = [[0 for x in range(w)] for y in range(h)]
score = 0 
rmse = 0
for i in range(start , end+ 1):
 # split into train and test sets
 look_back = i
 nxtrain = dff.iloc[0:train_size,:]
 nxtest = dff.iloc[train_size:len(dff),:]
 nytrain = dfl.iloc[0:train_size]
 nytest = dfl.iloc[train_size:len(dfl)]
 nxtrain, ytrain = create_dataset(nxtrain.values,nytrain,look_back)
 nxtest, ytest = create_dataset(nxtest.values,nytest.values,look_back)
 score , rmse = modelp(nxtrain, ytrain,nxtest,ytest,i,1,5)
 result[0+c][0]= i
 result[0+c][1]= score  
 result[0+c][2]= rmse
 c = c + 1 


# In[76]:


dfi = pd.DataFrame(np.array(result).reshape(end,3), columns = list("abc"))
dfi = dfi.rename(columns={'a': 'N', 'b': 'R2Score','c':'RMSE'})
print(dfi[:end-6])
print(dfi.plot(x=dfi["N"] , y=["RMSE"] , kind="bar", color = 'green'))
print("===================================================================================")
print(dfi.plot(x=dfi["N"] ,  y=["R2Score"] ,kind= 'bar'))


# 
#  **Can you use LSTM or other RNN models to predict the stock prices for a particular company for a continuous time period (e.g., the prices in the next five days)?     Show the true prices and predicted prices in the same chart.  **

# In[ ]:


def make_dataset(dataset,ydata,look_back):
    SEQUENCE_SIZE = look_back
    dataX, dataY = [], []
    for i in range(len(dataset)-SEQUENCE_SIZE-1):
        window = dataset[i:(i+SEQUENCE_SIZE)]
        after_window = ydata[i+SEQUENCE_SIZE+1:i+SEQUENCE_SIZE+SEQUENCE_SIZE+1]
        window = [[dataX] for dataX in window]
        dataX.append(window)
        dataY.append(after_window)
    return np.array(dataX), np.array(dataY)


# In[ ]:


# split into train and test sets
train_size = int(len(dff) * 0.70)
test_size = len(dff) - train_size
trainn = dff.iloc[0:train_size,:]
testn = dff.iloc[train_size:len(dff),:]
trainny = dfl.iloc[0:train_size]
testny = dfl.iloc[train_size:len(dfl)]


# Normalise

# In[ ]:


#Normalise
from scipy.stats import zscore  #method 1 
trainn = trainn.apply(zscore)
testn = testn.apply(zscore)


# In[80]:


trainny.head(15)


# In[ ]:


# create the feature dataset to be fed into the model
look_back = 5 #5 days
xtrain, ytrain = make_dataset(trainn.values,trainny,look_back)
xtest, ytest = make_dataset(testn.values,testny.values,look_back)


# In[ ]:


# reshape input to be [samples, time steps, features]
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], look_back))
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1],look_back))


# In[139]:


print("Training x data shape :",xtrain.shape)
print("Testing x data shape :",xtest.shape)
print("Training y data shape :",ytrain.shape)
print("Testing y data shape :",ytest.shape)


# In[141]:


ytest.shape


# In[ ]:


xtr= ytest
xtr  = [[x for x in xtr[y]]  for y in range(xtr.shape[0])]
resultstest = pd.DataFrame(xtr)
xtr= ytrain
xtr  = [[x for x in xtr[y]]  for y in range(xtr.shape[0])]
resultstrain = pd.DataFrame(xtr)


# In[95]:


results.isna().count()


# In[89]:


results.head()


# In[ ]:


ftrain = resultstrain.values
ftrain =np.nan_to_num(ftrain)
ftest =resultstest.values
ftest =np.nan_to_num(ftest)


# In[157]:


ftrain.shape
ftest.shape


# In[161]:


SEQUENCE_SIZE = look_back


# Define ModelCheckpoint outside the loop
checkpointlstm1 =  ModelCheckpoint(filepath="best_weightsfeature.hdf5", verbose=10, save_best_only=True,mode='max') # save best model

for i in range(2):
  print('Build model...')
  model4 = Sequential()
  model4.add(LSTM(100,input_shape=(SEQUENCE_SIZE, 5),return_sequences = True ))
  model4.add(Dropout(0.1))
  model4.add(Dense(32))
  model4.add(LSTM(100,return_sequences = False))
  model4.add(Dense(SEQUENCE_SIZE))
  model4.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
  monitor1 = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=10, mode='auto')
  print('Train...')
  callbacks_list1 = [monitor1,checkpointlstm1]
  model4.fit(xtrain,ftrain,validation_data=(xtest,ftest ),callbacks= callbacks_list1,verbose=2, epochs=100)  


# In[170]:



#model4.fit(xtrain,ftrain,validation_data=(,),callbacks= callbacks_list1,verbose=2, epochs=100)  
model4.load_weights("best_weightsfeature.hdf5") # load weights from best model
testPred = model4.predict(xtest,batch_size=5)
score = sk.metrics.r2_score(ftest , testPred)
print("Final R2 Score: {}".format(score))

print("RMSE:",math.sqrt(mean_squared_error(ftest , testPred)) )
print("\n")
#a = np.array(ftest ).reshape(ftest.shape[0],1).flatten()
chart_regression(testPred.flatten() ,ftest ,sort=True )

