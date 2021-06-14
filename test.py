# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:42:03 2021

@author: soheb
"""

import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Book1.xlsx")

df.head()


print(df.existence.value_counts())
print("Sum of nan:")
print(df.existence.isna().sum())

print(df.existence.value_counts())

#Replacing Yes with Y and No with N
df['existence'] = df['existence'].replace(['Yes','No'],['Y','N'])

print(df.existence.value_counts())

#Deleting nan value rows
df['existence'].dropna(inplace=True)
print("Sum of nan:")
print(df.existence.isnull().sum())

#train test split
train, test = train_test_split(df, test_size=0.2)


# remove special characters using the regular expression library
import re

#set up punctuations we want to be replaced
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

import preprocessor as p

# custum function to clean the dataset (combining tweet_preprocessor and reguar expression)
def clean_tweets(df):
  tempArr = []
  for line in df:
    # send to tweet_processor
    tmpL = p.clean(line)
    # remove puctuation
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
    tempArr.append(tmpL)
  return tempArr



# clean training data
train_tweet = clean_tweets(train["tweet"])
train_tweet = pd.DataFrame(train_tweet)


# append cleaned tweets to the training data
train["clean_tweet"] = train_tweet

# compare the cleaned and uncleaned tweets
print(train.head(10))


# clean the test data and append the cleaned tweets to the test data
test_tweet = clean_tweets(test["tweet"])
test_tweet = pd.DataFrame(test_tweet)
# append cleaned tweets to the training data
test["clean_tweet"] = test_tweet

# compare the cleaned and uncleaned tweets
print(test.tail())


train.dropna(inplace=True)

from sklearn.model_selection import train_test_split

# extract the labels from the train data
y = train.existence.values

# use 80% for the training and 20% for the test
x_train, x_test, y_train, y_test = train_test_split(train.clean_tweet.values, y,stratify=y,random_state=1,test_size=0.2, shuffle=True)



from sklearn.feature_extraction.text import CountVectorizer

# vectorize tweets for model building
vectorizer = CountVectorizer(binary=True, stop_words='english')

# learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(list(x_train) + list(x_test))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)




# classify using support vector classifier

from sklearn import svm

svm = svm.SVC(kernel = 'sigmoid', probability=True)

# fit the SVC model based on the given training data
prob = svm.fit(x_train_vec, y_train)

# perform classification and prediction on samples in x_test
y_pred_svm = svm.predict(x_test_vec)



from sklearn.metrics import accuracy_score
print("Accuracy score for SVC is: ", accuracy_score(y_test, y_pred_svm) * 100, '%') 







from sklearn.neighbors import KNeighborsClassifier
#Finding best n_neighbours for KNN
error = []

for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors=i)  
    knn.fit(x_train_vec, y_train)  
    pred_i = knn.predict(x_test_vec)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12,8))
plt.plot(range(1,60), error, color = 'red' , linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 10)
plt.title('Error Rate for K value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# classify using KNN classifier


knn = KNeighborsClassifier(n_neighbors=21)
  
knn.fit(x_train_vec, y_train)
  
# perform classification and prediction on samples in x_test
y_pred_knn = knn.predict(x_test_vec)


print("Accuracy score for KNN is: ", accuracy_score(y_test, y_pred_knn) * 100, '%')





from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

x_train_vec, y_train = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(x_train_vec[:,0], x_train_vec[:,1])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x_train_vec)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(x_train_vec)
plt.scatter(x_train_vec[:,0], x_train_vec[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()