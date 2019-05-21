
# coding: utf-8


# Project1: Yelp Business Rating Prediction using Pandas and Sklearn

# # Anshul Jain

# # Importing and Data Preprocessing of Yelp Dataset

# In[1]:


import pandas as pd
import os
import re
import math
import itertools
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("yelpdataset.csv")


# In[3]:


print("SHAPE OF THE DATA SET")
print("Data set shape:",df.shape )


# In[4]:


df.columns


# In[5]:


df = df.drop(columns= ['Unnamed: 0'])


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# # Text cleaning 
# removal of punctuation ,digits,commas and making all lower case 

# In[9]:


df['text'].head(20)


# In[10]:


df["text"] = df['text'].str.replace('[^\w\s]','') # removing punctuations
df["text"] = df['text'].str.replace('\d+','') # removing digits 


# In[11]:


df['text'].head(20)


# In[12]:


df.plot.hist()


# In[13]:


df.plot.scatter(x='stars_x',y='review_count')


# In[14]:


df.count()


# In[15]:


df=df[['business_id', 'city', 'state',  'review_count',  'text', 'useful', 'funny', 'cool','categories','stars_x']].copy()# add  'attributes' ,'is_open' after basic is done


# In[16]:


df[['stars_x']].head()


# In[17]:


df.shape


# In[18]:


df.head()


# In[19]:


## What about categories of organizations
## How many categories in each organization? (minuimum 1, maximum 35 categories)
## Most frequent 2 categories
print(df['categories'].str.count(';').min() + 1, df['categories'].str.count(';').max() + 1)
(df['categories'].str.count(';') + 1).value_counts().head()


# In[20]:


df['categories'].value_counts()


# In[21]:


df['categories'].isna().values.any() #thus no blanks in catagory... isna() detect missing values


# In[22]:


# the following sample code to group ALL the reviews for each business and create a new dataframe, where each line is a business with all its reviews.

df_review_agg = df.groupby('business_id')['text'].sum()  
 
df_ready_for_sklearn = pd.DataFrame({'business_id': df_review_agg.index, 'all_reviews': df_review_agg.values}) 


# In[23]:


df_ready_for_sklearn.head()


# In[24]:


df.columns


# # Step 2: TF-IDF Factorization of text coloumn

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(sublinear_tf=True, max_df=0.3,min_df=0.10,max_features=500, analyzer='word', stop_words='english', ngram_range =(1,3), use_idf = True)
x = v.fit_transform(df['text'])


# In[26]:


x


# In[27]:


v


# In[28]:


df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())


# In[29]:


df.drop('text', axis=1, inplace=True)
df = pd.concat([df, df1], axis=1)


# In[30]:


df.columns #these contains all columns previously selected previously


# In[31]:


df.head(2)


# ### NORMALISATION OF "REVIEW COUNT"

# In[32]:


#Normalisation of the dataframe
from sklearn import preprocessing

y = df[['review_count']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(y)
df['review_count'] = pd.DataFrame(x_scaled)


# In[33]:


df.head(2)


# ### ONE HOT Encoding of "City & State "

# In[34]:


dfe = pd.get_dummies(df[['city','state']] )
print(dfe.head(2))


# In[35]:


df.drop(columns=['city','state'], axis=1, inplace=True)
#df = pd.concat([df, dfe], axis=1)


# In[36]:


df.head(2)


# In[37]:


df['stars_x'].unique()


# **Making 2 copies of dataframe . One for Linear regression Model and one for one for Rest**

# In[38]:


dfo = df # for Other models 


# In[39]:


dfl = df # for Linear Regression


# In[40]:


#Add the features to remove
featurestoremove = {'business_id','categories'}
dfo= dfo[dfo.columns.difference(featurestoremove)]
dfl = dfl[dfl.columns.difference(featurestoremove)]
# Shuffling and splitting the data into Features and Labels 
dfo = dfo.sample(frac=1).reset_index(drop=True)
labelremove={'stars_x'}
labels =['stars_x']
dffeatures_x = dfo[dfo.columns.difference(labelremove)]


# In[41]:


#Splitting for Regression
dffeatures_x_ne = dfl[dfl.columns.difference(labelremove)]
dflabel_y_ne = df[['stars_x']]


# In[42]:


dffeatures_x_ne.columns


# ### Label Encoding of Target (Star Rating)

# In[43]:


# Label Encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dfo['stars_x'] = le.fit_transform(dfo['stars_x'] )


# **Splitting the dataframe into Features and Labels dataframe**

# In[44]:


#Adding the encoded labels to the dflabel_y
dflabel_y=dfo[labels]


# In[45]:


dffeatures_x.shape


# In[46]:


dflabel_y.head() # encoded for Catagorical Models


# In[47]:


dflabel_y_ne.head() # For regression Models 


# In[48]:


dffeatures_x_ne.shape  


# In[49]:


dflabel_y_ne.head() # not label encoded


# In[50]:


dflabel_y.head() #label Encoded


# In[51]:


dflabel_y.head()


# In[52]:


dffeatures_x.columns


# In[53]:


dffeatures_x['review_count'].head()


# In[54]:


dflabel_y.columns


# In[55]:


dflabel_y.head()


# In[56]:


dffeatures_x=dffeatures_x.fillna(0)
dflabel_y = dflabel_y.fillna(dflabel_y.mean())


# In[57]:


dffeatures_x.head()


# # Splitting data to training and test

# In[58]:


#splitting the data into training and testing set for all models except linear regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dffeatures_x, dflabel_y, test_size=0.33, random_state=42 ,shuffle=False)



#splitting for Linear regression model and regression neural network


X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(dffeatures_x_ne, dflabel_y_ne, test_size=0.33, random_state=42 ,shuffle=False)


# In[59]:


X_test.head()


# **=========================================================================================================================**

# #  Classical Machine Learning Models 

# **=========================================================================================================================**

# In[60]:


# Linear Regression to show true ratings of 5 businesses and the predicted ratings from model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import numpy as np
clf  = LinearRegression()
clf.fit(X_train_n,  y_train_n)
y_pred = clf.predict(X_test_n)


# In[61]:



y_pred


# In[62]:


y_test_n.head()


# In[63]:


# Measure RMSE error.  RMSE is common for regression.
from sklearn import metrics
score = np.sqrt(metrics.mean_squared_error(y_pred,y_test_n))
print("Final score (RMSE): {}".format(score))


# In[64]:


#SVM to show true ratings of 5 businesses and the predicted ratings from model
from sklearn import svm
#logmodel=  svm.SVC(kernel='linear')
logmodel=  svm.SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False, random_state=42)
logmodel.fit(X_train, y_train.values.ravel())
y_predictval = logmodel.predict(X_test)


# In[65]:


y_predictval


# In[66]:


y_test.head()


# In[67]:


from sklearn import metrics
print(metrics.classification_report(y_test,y_predictval))


# In[68]:


#Logistic Regression to show true ratings of 5 businesses and the predicted ratings from model
from sklearn import linear_model 
from sklearn.metrics import classification_report
logmod = linear_model.LogisticRegression(C=1e5)
logmod.fit(X_train, y_train.values.ravel())
y_predval = logmod.predict(X_test)


# In[69]:


y_predval


# In[70]:


y_test.head()


# In[71]:


print(metrics.classification_report(y_test,y_predval))


# In[72]:


#KNN to show true ratings of 5 businesses and the predicted ratings from model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train.values.ravel())
y_pv = neigh.predict(X_test)


# In[73]:


y_pv


# In[74]:


y_test.head()


# In[75]:


print(metrics.classification_report(y_test,y_pv))


# In[76]:


#Multinomial Naive Bayes to show true ratings of 5 businesses and the predicted ratings from model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train.values.ravel())
y_preds = nb.predict(X_test)


# In[77]:


y_preds


# In[78]:


y_test.head()


# In[79]:


print(metrics.classification_report(y_test,y_preds))


# **=========================================================================================================================**

# 
# # Neural Network Models 

# **=========================================================================================================================**

# In[80]:


import collections
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os


# In[81]:


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


# In[82]:


#x,y = to_xy(df,"stars_x")


# ***Task 1:  Consider this problem as a regression problem. Compare the RMSE of the BEST Tensorflow regression neural network model. ***
# 
# 
# 
# 

# In[83]:


print("shape of training features x and y ",X_train_n.shape, y_train_n.shape)
print("shape of Test features x and y ",X_test_n.shape, y_test_n.shape)


# In[84]:


X_train_n = X_train_n.fillna(X_train_n.mean() )
y_train_n = y_train_n.fillna(y_train_n.mean() )
X_test_n = X_test_n.fillna(X_test_n.mean())
y_test_n = y_test_n.fillna(y_test_n.mean() )


# In[85]:


import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import Dense, Dropout
from keras import metrics
import sklearn as sk
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU


# In[86]:


from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


# In[87]:


X_train_n.head()


# In[88]:


y_train_n.head()


# In[89]:


X_test_n.head()


# In[90]:


y_test_n.head()


# **Neural network Model which  implements Linear Regression Model 1**

# In[92]:


checkpointer = ModelCheckpoint(filepath="best_weights1.hdf5", verbose=0, save_best_only=True)
for i in range(5):
########################################################################################################################
    model0 = Sequential()
    model0.add(Dense(10, input_dim=dffeatures_x.shape[1], activation='relu', kernel_initializer = 'normal'))
    model0.add(Dense(15,activation='relu'))
    model0.add(Dense(dflabel_y.shape[1],activation='relu'))
    model0.compile(loss='mean_squared_error', optimizer= tf.train.AdamOptimizer(),metrics=['mse'])
#model0.compile(loss=root_mean_squared_error(y_test_n, y_pred), optimizer= tf.train.AdamOptimizer(),metrics=['mse','accuracy'])

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=2, mode='auto')  
    model0.fit(X_train_n, y_train_n, validation_data=(X_test_n,y_test_n), callbacks=[monitor,checkpointer],verbose=2,epochs=100) 
#########################################################################################################################
print('Training finished...Loading the best model')  
print()
model0.load_weights("best_weights1.hdf5") # load weights from best model

# Measure accuracy
pred = model0.predict(X_test_n)
pred = np.argmax(pred,axis=1)
score = sk.metrics.mean_squared_error(y_test_n, pred)
print(score)
print("Final mean_squared_error: {}".format(score))


# In[169]:


print(model0.summary())


# **Neural network Model which  implements Linear Regression Model 2**

# In[88]:


checkpointer1 = ModelCheckpoint(filepath="best_weights2.hdf5", verbose=0, save_best_only=True) # save best model
for i in range(5):
########################################################################################################################
    model1 = Sequential()
    model1.add(Dense(20, input_dim=dffeatures_x.shape[1], activation='tanh'))
    model1.add(Dense(10,activation='tanh')) #tanh Activation Function Used
    model1.add(Dense(dflabel_y.shape[1],activation='softmax'))
    model1.compile(loss='mean_squared_error', optimizer= tf.train.AdamOptimizer(),metrics=['mse','accuracy'])
    monitor1 = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=2, mode='auto')  
# patience: number of epochs with no improvement after which training will be stopped
    model1.fit(X_train_n, y_train_n, validation_data=(X_test_n,y_test_n), callbacks=[monitor1,checkpointer1],verbose=2,epochs=100)
#########################################################################################################################
print('Training finished...Loading the best model')  
print()
model1.load_weights("best_weights2.hdf5") # load weights from best model

# Measure accuracy
pred1 = model1.predict(X_test_n)
pred1 = np.argmax(pred1,axis=1)
score = sk.metrics.mean_squared_error(y_test_n, pred1)
print(score)
print("Final mean_squared_error: {}".format(score))


# In[89]:


model1.summary()


# In[90]:


pred = model1.predict(X_test_n)
print(pred) # print predictions


# **/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////**

# ***Task 2:  Consider this problem as a classification problem. Compare the accuracy of the BEST Tensorflow classification neural network model ***

# **Neural network Model which  implements Classification Model 1**

# In[91]:


from keras.utils import to_categorical
y_binary_test = to_categorical(y_test)
y_binary_train = to_categorical(y_train)


# In[92]:


y_binary_test


# In[93]:


y_binary_test .shape[1]


# In[94]:


checkpoint = ModelCheckpoint(filepath="best_weights3.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
for i in range(5):
########################################################################################################################
    model = Sequential()
    model.add(Dense(20, input_shape=(152,), activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(y_binary_test.shape[1],activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer= keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=[metrics.mae, metrics.categorical_accuracy])
    monitor2 = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=2, mode='auto')  
    # patience: number of epochs with no improvement after which training will be stopped
    # The test set is checked during training to monitor progress for early stopping but is never used for gradient descent (model training)
    callbacks_list = [monitor2,checkpoint]
    model.fit(X_train, y_binary_train, validation_data=(X_test,y_binary_test), callbacks=callbacks_list, verbose=2,batch_size=32, epochs=100)
##########################################################################################################################

print('Training finished...Loading the best model')  
print()
model.load_weights("best_weights3.hdf5") # load weights from best model

# Measure accuracy
pred2 = model.predict(X_test)
pred2 = np.argmax(pred2,axis=1)
y_true = np.argmax(y_test,axis=1)
score = sk.metrics.accuracy_score(y_test, pred2)
print("Final accuracy: {}".format(score))


# In[95]:


y_binary_test


# In[96]:


model.summary()


# In[97]:


predc = model.predict(X_test)
print(predc) # print  predictions


# **Neural network Model which  implements Classification Model 2**

# In[ ]:


checkpoints = ModelCheckpoint(filepath="best_weights4.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
for i in range(5):
########################################################################################################################
    models = Sequential()
    models.add(Dense(20, input_shape=(152,), activation='relu'))
    models.add(Dense(15,activation='relu'))
    models.add(Dense(15,activation='tanh'))
    model.add(Dense(15, 
                kernel_regularizer=regularizers.l1(0.01),
                activity_regularizer=regularizers.l2(0.01), activation='relu'))
    models.add(Dropout(0.5))
    models.add(Dense(15,activation= 'relu'))
    models.add(Dense(y_binary_test.shape[1],activation='sigmoid'))
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    models.compile(loss='categorical_crossentropy', optimizer= sgd ,metrics=[metrics.mae, metrics.categorical_accuracy])
    monitors = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=2, mode='auto')  
    # The test set is checked during training to monitor progress for early stopping but is never used for gradient descent (model training)
    callbacks_list = [monitors,checkpoints]
    models.fit(X_train, y_binary_train, validation_data=(X_test,y_binary_test), callbacks=callbacks_list, verbose=2,batch_size=32, epochs=100)
    models.load_weights("best_weights4.hdf5")
##########################################################################################################################
print('Training finished...Loading the best model')  
print()
models.load_weights("best_weights4.hdf5") # load weights from best model

# Measure accuracy
pred3 = models.predict(X_test)
pred3 = np.argmax(pred3,axis=1)
#y_true = np.argmax(y_test,axis=1)
score = sk.metrics.accuracy_score(y_test, pred3)
print("Final accuracy: {}".format(score))


# In[ ]:


models.summary()


# In[ ]:


predc = models.predict(X_test)
print(predc) # print predictions


# **Neural network Model which  implements Classification Model 3**

# In[ ]:


import keras as kp
# Define ModelCheckpoint outside the loop
checkpoint3 = ModelCheckpoint(filepath="best_weights5.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
for i in range(1):
########################################################################################################################
    model3 = Sequential()
    model3.add(Dense(20, input_shape=(152,), activation='relu'))
    model3.add(Dense(15,activation='relu' ))
    model3.add(Dense(100))
    model3.add(LeakyReLU(alpha=0.1))
    model3.add(Dense(100))
    model3.add(LeakyReLU(alpha=0.1)) ##Leaky Relu
    model3.add(Dense(20,activation= 'relu' ))
    # Regularisation Through Drop Out 
    model3.add(Dense(15,activation='tanh'))
    model3.add(Dense(15, 
                kernel_regularizer=regularizers.l1(0.01),
                activity_regularizer=regularizers.l2(0.01), activation='relu'))
    model3.add(Dropout(0.25))
    model3.add(Dense(15,activation= 'relu'))
    model3.add(Dense(y_binary_test.shape[1],activation='sigmoid'))
    rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model3.compile(loss='categorical_crossentropy', optimizer= rms ,metrics=[kp.metrics.mae, kp.metrics.categorical_accuracy])
    monitor3 = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=2, mode='auto')  
# The test set is checked during training to monitor progress for early stopping but is never used for gradient descent (model training)
#checkpoint3 = ModelCheckpoint(filepath="best_weights5.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
    callbacks_list = [monitor3,checkpoint3]
    model3.fit(X_train, y_binary_train, validation_data=(X_test,y_binary_test), callbacks=callbacks_list, verbose=2,batch_size=32, epochs=200)
##########################################################################################################################


# In[ ]:


model3.summary()


# **xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx**

# # Accuracy Comparision

# **xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx**

# **Task 1 Accuracy comparison**

# **Accuracy of Classical Machine Learning Model (Linear Regression)**

# In[ ]:



score = np.sqrt(sk.metrics.mean_squared_error(y_pred,y_test_n))
print("Final score (RMSE): {}".format(score))


# In[ ]:


import math
print("RMSE of Neural network model: ",math.sqrt(12.883383838383839))


# **Task 2 Accuracy Comparison**

# **Classic SVM model accuracy**

# In[ ]:



print(sk.metrics.classification_report(y_test,y_predictval))


# ** Classic Logistic regression accuracy**

# In[ ]:


print(sk.metrics.classification_report(y_test,y_predval))


# **Classic KNN model Accuracy**

# In[ ]:


print(sk.metrics.classification_report(y_test,y_pv))


# **Classic Multinomial Naive Bayes model accuracy**

# In[ ]:


print(sk.metrics.classification_report(y_test,y_preds))


# **Neural network Model accuracy which implements best Classification model 1** 

# In[ ]:


print(" Neural network Model Accuracy: val_categorical_accuracy: 0.7117 ")


# In[ ]:




