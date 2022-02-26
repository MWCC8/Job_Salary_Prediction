#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:23:27 2022

@author: michaelchurchcarson
"""

##### INSY 669 Individual Assignment

# February 25, 2022

# Mihael Church Carson (260683849)

#####------------Predicting the Saleries for Job Descriptions------------#####

# Import Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Import Data

Train_rev1 = pd.read_csv('Train_Rev1.csv')

Train_rev1 = Train_rev1.dropna()

# Randomly select 2500 data points from the imported dataset

df = Train_rev1.sample(n = 2500, random_state = 0)

# =============================================================================
# # Question 1
# =============================================================================


# Create classification model to predict high (75th percentile and above) or low (below 75th percentile) salary 
# from the text contained in the job descriptions.

# Create a new column with salary percentile
df['Salary_Percentile'] = df['SalaryNormalized'].rank(pct = True)


# Create a new columnwith a value of 1 if the percentile is equal to or above 0.75, and 0 otherwise.

df['Salary'] = np.where(df['Salary_Percentile'] >=0.75, 'high', 'low')

# Create variable out of FullDescription column

word_tokenizer=RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_nltk=set(stopwords.words('english'))
temp = df['FullDescription']
def tokenize_text(version_desc):
    lowercase=version_desc.lower()
    text = wordnet_lemmatizer.lemmatize(lowercase)
    tokens = word_tokenizer.tokenize(text)
    return tokens

vect = TfidfVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
total_features_words = vect.fit_transform(df['FullDescription'])
total_features_words = total_features_words.toarray()

# Specify the X and Y variables for the classification model

X = total_features_words
y = df['Salary']

# Seperate the 2500 data points into training (80%) and test (20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model on training set

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the testing set

y_pred = gnb.predict(X_test)

# Compare actual response values (y_test) with predicted response values (y_pred)

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

# Confusion matrix

matrix = confusion_matrix(y_test,y_pred, labels=['high','low'])

print('Confusion matrix : \n',matrix)

matrix_df = pd.DataFrame(matrix)
#matrix_df.to_csv('ConfusionMatrix.csv')

# Identifying the top 10 words for high and low salaries

# Create a dataframe for high paying jobs
df_high = df[df['Salary'] == 'high']

# Create a dataframe for low paying jobs
df_low = df[df['Salary'] == 'low']

# Find the top 10 words for high salaries

feature_array = np.array(vect.get_feature_names())

response_high = vect.transform(df_high['FullDescription'])

tfidf_sorting_high = np.argsort(response_high.toarray()).flatten()[::-1]

                           
n = 10
top_n_high = feature_array[tfidf_sorting_high][:n]
top_n_high

# Find the top 10 words for low salaries

feature_array = np.array(vect.get_feature_names())

response_low = vect.transform(df_low['FullDescription'])

tfidf_sorting_low = np.argsort(response_low.toarray()).flatten()[::-1]

                           
n = 10
top_n_low = feature_array[tfidf_sorting_low][:n]
top_n_low


# =============================================================================
# # Question 2 
# =============================================================================

# create a dataframe that has more predictor variables from the the dataset
total_features_words = pd.DataFrame(total_features_words)

df1 = df[['ContractType', 'ContractTime', 'Category']]

# One hot encoding for the binray categorical variables contract type and contract time
df1['Contract'] = np.where(df1['ContractType'] == 'full_time', 1, 0)
df1['Time'] = np.where(df1['ContractTime'] == 'permanent', 1, 0)

# Add new columns to total_features_words
total_features_words['ContractType'] = df1['Contract'].values
total_features_words['ContractTime'] = df1['Time'].values


# Change category into a category variable and then create cat codes
df1["Category"] = df1["Category"].astype('category')
df1["Category"] = df1["Category"].cat.codes

# Add new columns to total_features_words
total_features_words['Category'] = df1['Category'].values

# Specify the X and Y variables for the classification model

X = total_features_words

y = df['Salary']


# Seperate the 2500 data points into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Train the model on training set
gnb2 = GaussianNB()
gnb2.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = gnb2.predict(X_test)

# Compare actual response values (y_test) with predicted response values (y_pred)

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

matrix2 = confusion_matrix(y_test,y_pred, labels=['high','low'])

print('Confusion matrix : \n',matrix2)

matrix2_df = pd.DataFrame(matrix)
#matrix2_df.to_csv('ConfusionMatrix2.csv')
