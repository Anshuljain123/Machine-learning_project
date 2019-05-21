
# coding: utf-8

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


df = pd.read_csv("network_intrusion_data.csv")


# In[3]:


print("SHAPE OF THE DATA SET")
print("Data set shape:",df.shape )


# In[4]:


df = df[:5000]


# In[5]:


df


# In[6]:


df.columns


# In[7]:


df.columns = [
'duration',
'protocol_type',
'service',
'flag',
'src_bytes',
'dst_bytes',
'land',
'wrong_fragment',
'urgent',
'hot',
'num_failed_logins',
'logged_in',
'num_compromised',
'root_shell',
'su_attempted',
'num_root',
'num_file_creations',
'num_shells',
'num_access_files',
'num_outbound_cmds',
'is_host_login',
'is_guest_login',
'count',
'srv_count',
'serror_rate',
'srv_serror_rate',
'rerror_rate',
'srv_rerror_rate',
'same_srv_rate',
'diff_srv_rate',
'srv_diff_host_rate',
'dst_host_count',
'dst_host_srv_count',
'dst_host_same_srv_rate',
'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate',
'dst_host_serror_rate',
'dst_host_srv_serror_rate',
'dst_host_rerror_rate',
'dst_host_srv_rerror_rate',
'outcome'
]


# In[8]:


df.columns


# In[9]:


print("SHAPE OF THE DATA SET")
print("Data set shape:",df.shape )


# In[10]:


df.info()


# In[11]:


df.shape


# In[12]:


df.describe()


# In[13]:


df.count()


# In[14]:


df.isna()


# In[15]:


# deleting missing rows
df.dropna(axis=0, how='all', inplace=False) 


# In[16]:


df.plot.hist()


# In[17]:


df.columns


# In[18]:


df._get_numeric_data().columns


# In[19]:


# one hot encoding
df_ohe = pd.get_dummies(df[['protocol_type', 'service', 'flag','outcome']] )
print(df.head())


# In[20]:


df.drop(columns=['protocol_type', 'service', 'flag','outcome'], axis=1, inplace=True)
df = pd.concat([df, df_ohe], axis=1)


# In[21]:


df.head()


# In[22]:


df.describe


# In[23]:


#Normalisation of the dataframe
from sklearn import preprocessing

y = df[[ 'src_bytes']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(y)
df[ 'src_bytes'] = pd.DataFrame(x_scaled)


# In[24]:


#Normalisation of the dataframe
from sklearn import preprocessing

y = df[[ 'count']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(y)
df[ 'count'] = pd.DataFrame(x_scaled)


# In[25]:


df


# In[26]:


#Normalisation of the dataframe
from sklearn import preprocessing

y = df[[ 'dst_bytes']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(y)
df[ 'dst_bytes'] = pd.DataFrame(x_scaled)


# In[27]:


#Normalisation of the dataframe
from sklearn import preprocessing

y = df[[ 'dst_host_count']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(y)
df[ 'dst_host_count'] = pd.DataFrame(x_scaled)


# In[28]:


#Normalisation of the dataframe
from sklearn import preprocessing

y = df[[ 'dst_host_srv_count']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(y)
df[ 'dst_host_srv_count'] = pd.DataFrame(x_scaled)


# In[29]:


#Normalisation of the dataframe
from sklearn import preprocessing

y = df[[ 'srv_count']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(y)
df[ 'srv_count'] = pd.DataFrame(x_scaled)


# In[30]:


df.head()


# In[31]:


# merging of df with df_ohe
#df = pd.concat([df, df_ohe], axis=1)


# In[32]:


df.head()


# In[33]:


#df.drop(columns=[ 'protocol_type', 'service', 'flag'], axis=1, inplace=True)


# In[34]:


df.head()


# In[35]:


#df= df[df.columns.difference(featurestoremove)]
df = df.sample(frac=1).reset_index(drop=True)
#labelremove={'outcome_buffer_overflow.','outcome_loadmodule.','outcome_neptune.','outcome_normal.','outcome_perl.','outcome_smurf.'}
labelremove={'outcome_normal.'}
#labels =['outcome_buffer_overflow.','outcome_loadmodule.','outcome_neptune.','outcome_normal.','outcome_perl.','outcome_smurf.']
labels = ['outcome_normal.']
#df_x = df[df.columns.difference(labelremove)]


# In[36]:


#Splitting for x and y
df_x = df[df.columns.difference(labelremove)]
#df_y = df[['outcome_buffer_overflow.','outcome_loadmodule.','outcome_neptune.','outcome_normal.','outcome_perl.','outcome_smurf.']]
df_y = df[['outcome_normal.']]


# In[37]:


df_x


# In[38]:


df_x.shape


# In[39]:


df_y.shape


# In[40]:


df.columns


# In[41]:


df_y.head()


# In[42]:


#df_x=df_x.fillna(0)
#df_y = df_y.fillna(df_y.mean())


# In[43]:


#df


# In[44]:


#splitting the data into training and testing set for all models except linear regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42 ,shuffle=False)


# In[45]:


X_test.transpose


# In[46]:


df_x.shape


# In[47]:


y_train.shape


# In[48]:


y_test.shape


# In[49]:


#numpy.reshape()


# In[50]:


y_test.head()


# # Models training

# In[51]:


#SVM 
from sklearn import svm
#logmodel=  svm.SVC(kernel='linear')
logmodel=  svm.SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False, random_state=42)
logmodel.fit(X_train, y_train.values.ravel())
y_svm = logmodel.predict(X_test)


# In[52]:


y_svm


# In[53]:


y_test.head()


# In[54]:


from sklearn.metrics import confusion_matrix
y_test
y_svm
confusion_matrix(y_test, y_svm, labels=None, sample_weight=None)


# In[55]:


from sklearn import metrics
print(metrics.classification_report(y_test,y_svm))


# In[56]:


#Logistic Regression 
from sklearn import linear_model 
from sklearn.metrics import classification_report
logmod = linear_model.LogisticRegression(C=1e5)
logmod.fit(X_train, y_train.values.ravel())
y_logistic = logmod.predict(X_test)


# In[57]:


y_logistic


# In[58]:


y_test.head()


# In[59]:


print(metrics.classification_report(y_test,y_logistic))


# In[60]:


#KNN 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train.values.ravel())
y_knn = neigh.predict(X_test)


# In[61]:


y_knn


# In[62]:


y_test.head()


# In[63]:


print(metrics.classification_report(y_test,y_knn))


# In[64]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train.values.ravel())
y_gaussian = nb.predict(X_test)


# In[65]:


y_gaussian


# In[66]:


y_test.head()


# In[67]:


print(metrics.classification_report(y_test,y_gaussian))


# # Neural network

# In[68]:


import collections
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os


# In[69]:


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


# In[70]:


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


# In[71]:


from keras.utils import to_categorical
y_binary_test = to_categorical(y_test)
y_binary_train = to_categorical(y_train)


# In[72]:


checkpoint = ModelCheckpoint(filepath="best_weights3.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
for i in range(2):
########################################################################################################################
    model = Sequential()
    model.add(Dense(20, input_shape=(59,), activation='relu'))
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


# In[73]:


model.summary()


# In[74]:


import keras as kp
# Define ModelCheckpoint outside the loop
checkpoint3 = ModelCheckpoint(filepath="best_weights5.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
for i in range(1):
########################################################################################################################
    model3 = Sequential()
    model3.add(Dense(20, input_shape=(59,), activation='relu'))
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


# In[75]:


model3.summary()


# # CNN

# In[103]:


import pandas as pd 
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sklearn as sk
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import Dense, Dropout , Flatten
import sklearn as sk
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D


# In[104]:


X_train.shape


# In[105]:


xtrain= X_train[:10]


# In[106]:


xtrain.shape


# In[107]:


xtest = X_test[:10]


# In[108]:


xtest.shape


# In[109]:


ytrain=y_train[:10]


# In[110]:


ytrain.shape


# In[111]:


ytest = y_test[:10]


# In[112]:


ytest.shape


# In[113]:


xtr =  xtrain.values
xts  =  xtest.values


# In[114]:


x = xtr[:7]


# In[115]:


xt = xts[:7]


# In[116]:


x = x.reshape((7, 1, 59, 1)) # value you choose, 1 row, 59 colomns, 1 greyscale
xt = xt.reshape((7,1,59,1))


# In[117]:


x.shape


# In[118]:


xt.shape


# In[119]:


#encoding of output y (necessary)
ytrn = keras.utils.to_categorical(ytrain)
ytst= keras.utils.to_categorical(ytest)


# In[120]:


print(ytrn.shape)
print(ytst.shape)


# In[121]:


ytrn = ytrn[:7]
ytst = ytrn[:7]


# In[122]:


ytrn.shape


# In[123]:


ytst.shape


# In[124]:


num_classes = 2


# In[127]:


from keras.optimizers import Adam
# Define ModelCheckpoint outside the loop
checkpoint3 = ModelCheckpoint(filepath="best_weights5.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
for i in range(3):
########################################################################################################################
    model = Sequential()
    model.add(Conv2D(32, (1, 1), padding='same',input_shape=(1, 59, 1)))

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

    model.add(Dense(num_classes, activation="softmax"))

    monitor3 = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=2, mode='auto')  
    

# show not only log loss but also accuracy for each epoch using metrics=['accuracy']

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
# The test set is checked during training to monitor progress for early stopping but is never used for gradient descent (model training)
#checkpoint3 = ModelCheckpoint(filepath="best_weights5.hdf5", verbose=0, save_best_only=True,mode='max') # save best model
    callbacks_list = [monitor3,checkpoint3]

    model.fit(x, ytrn, validation_data=(xt,ytst), callbacks=callbacks_list, verbose=2,batch_size=32, epochs=100)

