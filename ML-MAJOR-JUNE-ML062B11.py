#!/usr/bin/env python
# coding: utf-8

# # major project

# In[3]:


import pandas as pd #linear algebra
import numpy as np #data processing, CSV file I/O
import re
import string
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=10,8
matplotlib.get_backend()
df = pd.read_csv('Information.csv',encoding='latin1') 


# # reading the data

# In[4]:


df.head(2)


# In[5]:


df.columns


# In[6]:


#taking only required number of columns for data analysis
data = pd.read_csv("Information.csv",usecols= [0,5,19,17,21,10,11],encoding='latin1')


# In[7]:


data.head(2)


# # cleaning the data

# In[8]:


#removing genders other than male and female


# In[9]:


df = df[(df['gender'] == 'male') | (df['gender'] == 'female')]


# In[10]:



# LABEL ENCODING _golden, _unit_state
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['_golden2'] = le.fit_transform(df['_golden'])
df['_unit_state2'] = le.fit_transform(df['_unit_state'])
df['gender2'] = le.fit_transform(df['gender'])
df['_unit_state2'] = le.fit_transform(df['_unit_state'])


# In[11]:


#removing unwanted column from the dataset
df.drop(labels=['_unit_id'],axis=1,inplace=True)


# # questions asked on the dataset 

# # Question1 What are the most common emotions/words used by Males and Females?
# 

# In[12]:


#cleaning the data 
#splitting the sentences to inidividual words
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s

data['Tweets'] = [cleaning(s) for s in data['text']]
data['Description'] = [cleaning(s) for s in data['description']]


# In[13]:


data.head(2)


# In[14]:


data = data[(data['gender'] == 'male') | (data['gender'] == 'female')]


# In[15]:


data.gender.value_counts()


# In[16]:


Male = data[data['gender'] == 'male']
Female = data[data['gender'] == 'female']
Male_Words = pd.Series(' '.join(Male['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]
Female_Words = pd.Series(' '.join(Female['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]


# In[17]:


Female_Words


# In[18]:


#As you can see 'and' word is most commonly used by female if you exclude ' 'character
Female_Words.plot(kind='bar',stacked=True, colormap='plasma')


# In[19]:


Male_Words


# In[20]:



Male_Words.plot(kind='bar',stacked=True, colormap='plasma')

#A bar graph is plotted acoording to the usage of words
#As you can see 'the' word is most commonly used word by males if you exclude ' 'character


# # Question 2 What is the highest tweet_count got by a male and female?

# In[25]:


Male = df[df['gender'] == 'male']
Female = df[df['gender'] == 'female']
df[Male.notnull()][['gender:confidence','gender','tweet_count']].sort_values('gender',ascending=False).head(1)
#for male


# In[26]:


df[Female.notnull()][['gender:confidence','gender','tweet_count']].sort_values('gender',ascending=False).head(1)
#for female


# # Calculating gender count in the dataset

# In[67]:



df['gender'].value_counts()


# In[68]:


df.info()


# # To check the correlation between columns

# In[ ]:





# In[69]:


import seaborn as sb
sb.heatmap(df.corr(), annot=True)


# In[70]:


df.describe()


# In[71]:


df.columns


# In[72]:


df.describe().columns
df.dtypes


# # Ensemble Machine learning Modelling (3 Classification Algorithms)

# #Dependant variable is 'gender'
# 
# #Independant variables are 'tweet_id','retweet_count', 'tweet_count'

# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


X = df[['tweet_id','retweet_count', 'tweet_count']].values
Y = df[['gender']].values


# In[75]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


# In[76]:


X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# # 1. K Nearest Neighbor(KNN) 

# The k-nearest-neighbors algorithm is a classification algorithm, and it is supervised: it takes a bunch of labelled points and uses them to learn how to label other points. To label a new point, it looks at the labelled points closest to that new point (those are its nearest neighbors), and has those neighbors vote, so whichever label the most of the neighbors have is the label for the new point (the “k” is the number of neighbors it checks).

# In[77]:


#importing the algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[78]:


#here we compute KNN algorithm for value 1
knn = KNeighborsClassifier(n_neighbors=1)


# In[79]:


knn


# In[80]:


knn.fit(X_train, Y_train.ravel())


# In[81]:


#taking the predicting value with X_test
Y_pred = knn.predict(X_test)


# In[82]:


from sklearn.metrics import accuracy_score
#to calculate the accuracy of the algorithm


# # Accuracy

# In[83]:


accuracy_score(Y_test, Y_pred)*100
#here we can see we got the accuracy of 51%


# # 2.Logistic Regression (Predictive Learning Model) :

# Logistic regression is a technique in statistical analysis that attempts to predict a data value based on prior observation. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes). The goal of logistic regression is to find the best fitting model to describe the relationship between the dichotomous characteristic of interest (dependent variable = response or outcome variable) and a set of independent (predictor or explanatory) variables.

# In[84]:


from sklearn.linear_model import LogisticRegression
#importing the algorithm


# In[85]:


LogReg = LogisticRegression()


# In[86]:


LogReg.fit(X_train, Y_train.ravel())


# # Accuracy

# In[87]:


LogReg.score(X_test,Y_test)*100
#Here we got the accuracy of 51%


# # 3.Decision Trees

# A decision tree is a graphical representation of specific decision situations that are used when complex branching occurs in a structured decision process. It breaks down a data set into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node has two or more branches and a leaf node represents a classification or decision. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.

# In[88]:


from sklearn.tree import DecisionTreeClassifier


# In[89]:


dt=DecisionTreeClassifier(criterion='gini')
#Gini Index — The Gini coefficient measures the inequality among values of a distributed groups. 
#It takes values from 0–1. A Gini coefficient of zero expresses perfect equality where all values are the same . 
#A Gini coefficient of one \ expresses maximal inequality among values.


# In[90]:


dt.fit(X_train,Y_train)


# In[91]:


actual=Y
predict=dt.predict(X)


# In[92]:


from sklearn import metrics


# # Accuracy

# In[93]:


print(metrics.accuracy_score(actual,predict)*100)
#here we can see we got the accuracy of 84%


# # Conclusion

# # After applying Train/Test split method and comparing the three classification models i.e Logistic algorithm ,KNN(n=1) and Decision Trees , we conclude that Decision Trees algorithm suits best with 84% accuracy and the Logistic with next highest accuracy of 51.6% and last KNN  with 51.3% for k=1

# However, since we only used train split method for accuracy calculation, we got decision trees with highest accuracy 

# However, if we use other methods like cross-validation or training model with entire data set we get different accuracies

# and different algorithm model with highest accuracy 

# In[ ]:




