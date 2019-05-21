
# coding: utf-8


# # Project: Yelp Business Rating Prediction using Pandas and Sklearn


# Steps of the Project 
# 1. Import the processed dataset (dataset of Business and review joined together and 15000 rows are extracted)
#    (select 5000 rows from 15000 and also including city and state )
# 2. Preprocess and analyse and select the appropriate features fromt the data 
# 3. TDIDF the Text Column 
# 4. Encode the features and labels ( features -one hot encoding , Label - Label encoding)   
# 5. Normalization of the data
# 6. Split the data into Training and Testing set 
# 7. Train the models
# 8. Test the Model against known data.
# 

# # Step 1 & 2  : Importing and Data Preprocessing of Yelp Dataset

# In[123]:


import pandas as pd
import os
import re
import math
import itertools
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[124]:


df = pd.read_csv("yelpdataset.csv")


# In[125]:


print("SHAPE OF THE DATA SET")
print("Data set shape:",df.shape )


# In[126]:


df.columns


# In[127]:


df.head()


# In[128]:


df = df.drop(columns= ['Unnamed: 0'])


# In[129]:


df


# In[130]:


df.shape


# In[131]:


df.info()


# In[132]:


df.describe()


# In[133]:


df.columns


# In[134]:


df.tail(2)


# In[135]:


df.plot.hist()


# In[136]:


df.plot.scatter(x='stars_x',y='review_count')


# In[137]:


df.index


# In[138]:


df.count()


# In[139]:


dfif=df[['business_id', 'city', 'state',  'review_count',  'text', 'useful', 'funny', 'cool','categories','stars_x']].copy()# add  'attributes' ,'is_open' after basic is done


# In[140]:


df.to_csv("test.csv")


# In[141]:


dfif.shape


# In[142]:


dfif.head()


# In[143]:


#df = df.sample(frac=1).reset_index(drop=True)
dfi = dfif[0:5000].sample(frac=1).reset_index(drop=True)


# In[144]:


dfi


# In[145]:


dfi.shape


# In[146]:


dfi = dfi.sample(frac=1).reset_index(drop=True)


# In[147]:


dfi


# In[148]:


dfi.head()


# In[149]:


dfi['cool'].value_counts().plot.hist() 


# In[150]:


dfi['categories'].value_counts().plot.hist()


# In[151]:


dfi.describe


# In[152]:


dfi.describe(include= 'all')


# In[153]:


## What about categories of organizations
## How many categories in each organization? (minuimum 1, maximum 35 categories)
## Most frequent 2 categories
print(dfi['categories'].str.count(';').min() + 1, dfi['categories'].str.count(';').max() + 1)
(dfi['categories'].str.count(';') + 1).value_counts().head()


# In[154]:


## How many categories we have?
categories = pd.concat(
    [pd.Series(row['business_id'], row['categories'].split(';')) for _, row in dfi.iterrows()]
).reset_index()
categories.columns = ['categorie', 'business_id']
#categories.head(10)


# In[155]:


categories.tail()


# In[156]:


categories.head(10)


# In[157]:


## How many categories which are unique?
print(categories['categorie'].nunique())


# In[158]:


## Most frequent categories(Top 10)
categories['categorie'].value_counts().head(10)


# In[159]:


fig, ax = plt.subplots(figsize=[5,10])
sns.countplot(data=categories[categories['categorie'].isin(
    categories['categorie'].value_counts().head(25).index)],
                              y='categorie', ax=ax)
plt.show()


# In[160]:


categories_ = categories[
    (categories['categorie'].isin(categories['categorie'].value_counts().head(25).index))
]
ct = pd.crosstab(
    categories_['business_id'],
    categories_['categorie'])

fig, ax = plt.subplots(figsize=[10,10])
sns.heatmap(ct.head(25), ax=ax, cmap='Blues')
ax.set_title('Top 25-cat, Random 25 organizations')


# In[161]:


dfi['categories'].value_counts()


# In[162]:


dfi['categories'].isna().values.any() #thus no blanks in catagory... isna() detect missing values


# In[163]:


# the following sample code to group ALL the reviews for each business and create a new dataframe, where each line is a business with all its reviews.

df_review_agg = dfi.groupby('business_id')['text'].sum()  
 
df_ready_for_sklearn = pd.DataFrame({'business_id': df_review_agg.index, 'all_reviews': df_review_agg.values}) 


# In[164]:


df_ready_for_sklearn.head()


# # Step 2: TF-IDF Factorization of text coloumn

# In[165]:


from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(sublinear_tf=True, max_df=0.3,min_df=0.10,max_features=500, analyzer='word', stop_words='english', ngram_range =(1,3), use_idf = True)
x = v.fit_transform(dfi['text'])


# In[166]:


x


# In[167]:


v


# In[168]:


df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
print(df1.head())


# In[169]:


df1.shape


# In[170]:


dfi.columns


# In[171]:


df1.columns


# In[172]:


data = pd.concat([dfi, df1], axis=1)


# In[173]:


data.shape


# In[174]:


data.head()


# In[175]:


#count of Not NA values
data.isna().count()


# In[176]:


#printing business with corresponding rating
data[['business_id','stars_x']].head()


# # Step 3:One- hot encoding of categorical values coloumn 'categories' and
# # Label encoding of Label  "stars_x"

# In[177]:


#pd.get_dummies(dfi,'categories')
data= pd.get_dummies(data, columns=['city','state','categories'], drop_first=True )


# In[178]:


# Label Encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data1 = data.copy()
data1['stars_x'] = le.fit_transform(data['stars_x'] )


# # The Column Features that is taken into account

# In[179]:


print(data.columns)


# In[180]:


data.shape


# In[181]:


#Add the features to remove
featurestoremove = {'text','business_id'}
dfinum = data[data.columns.difference(featurestoremove)]

# Shuffling and splitting the data into Features and Labels 
dfinum = dfinum.sample(frac=1).reset_index(drop=True)
labelremove={'stars_x'}
labels =['stars_x']
dffeatures_x = dfinum[dfinum.columns.difference(labelremove)]
dflabel_y=data1[labels]


# In[182]:


dffeatures_x.shape


# In[183]:


dffeatures_x.describe()


# In[184]:


dffeatures_x.columns


# In[185]:


dffeatures_x['review_count'].head()


# In[186]:


dflabel_y.columns


# In[187]:


dflabel_y.head()


# In[188]:


dffeatures_x=dffeatures_x.fillna(0)
dflabel_y = dflabel_y.fillna(dflabel_y.mean())


# In[189]:


dffeatures_x.head()


# In[190]:


dffeatures_x.dtypes


# In[191]:


dflabel_y.shape


# In[192]:


dffeatures_x.head()


# In[193]:


dffeatures_x.to_csv('test')


# # Step 5: Feature Normalization of the Dataframe 

# In[194]:


#Normalisation of the dataframe
#from sklearn import preprocessing

#y = dffeatures_x.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(y)
#dffeatures_x = pd.DataFrame(x_scaled)


# # We take many other Features( Other than Text ,Review Count , Star Rating)#
# 1. City 
# 2. State 
# 3. useful       
# 4. funny           
# 5. cool           
# 6. Catagory

# In[195]:


dffeatures_x.head(5)


# In[196]:


#adding Business id and rating to the Features
dfinum.columns


# In[197]:


import tensorflow as tf


# In[225]:


df.columns


# In[199]:


review = tf.feature_column.numeric_column('review_count')


# In[226]:


#business = tf.feature_column.numeric_column('business_id')
cat = tf.feature_column.categorical_column_with_hash_bucket("categories", hash_bucket_size= 1000)


# In[227]:



feat_cols = [review, cat]


# # Input function for estimator object( play around with batch size and num_epochs)

# In[228]:


input_func = tf.estimator.inputs.pandas_input_fn (x=X_train, y = y_train, batch_size = 10, num_epochs= 1000, shuffle = True)


# # create the estimator model. Use a DNN regressor. play around with hidden units

# In[232]:


#model = tf.estimator.DNNRegressor(hidden_units = [2,2,2], feature_columns= feat_cols)mo
model = tf.estimator.LinearClassifier( feature_columns= feat_cols)


# # Train the moel for 1000 steps (later can be trained for more)

# In[233]:


model.train(input_fn = input_func, steps = 1000)


# # Step 6 :Splitting data to training and test

# In[205]:


#splitting the data into training and testing set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dffeatures_x, dflabel_y, test_size=0.33, random_state=42 ,shuffle=False)


# In[206]:


X_test.head()


# # Step 7 and 8 : Models Implementation & Testing

# In[207]:


# Linear Regression to show true ratings of 5 businesses and the predicted ratings from model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import numpy as np
clf  = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test.head())


# In[208]:


np.floor(y_pred)


# In[209]:


y_test.head()


# In[210]:


#SVM to show true ratings of 5 businesses and the predicted ratings from model
from sklearn import svm
logmodel=  svm.SVC()
logmodel.fit(X_train, y_train.values.ravel())
y_predictval = logmodel.predict(X_test.head())


# In[211]:


y_predictval


# In[212]:


y_test.head()


# In[213]:


#Logistic Regression to show true ratings of 5 businesses and the predicted ratings from model
from sklearn import linear_model 
from sklearn.metrics import classification_report
logmod = linear_model.LogisticRegression(C=1e5)
logmod.fit(X_train, y_train.values.ravel())
y_predval = logmod.predict(X_test.head())


# In[214]:


y_predval


# In[215]:


y_test.head()


# In[216]:


#KNN to show true ratings of 5 businesses and the predicted ratings from model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train.values.ravel())
y_pv = neigh.predict(X_test.head())


# In[217]:


y_pv


# In[218]:


y_test.head()


# In[219]:


#Multinomial Naive Bayes to show true ratings of 5 businesses and the predicted ratings from model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train.values.ravel())
y_preds = nb.predict(X_test.head())


# In[220]:


y_preds


# In[221]:


y_test.head()

