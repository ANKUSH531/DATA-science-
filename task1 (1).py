#!/usr/bin/env python
# coding: utf-8

#   
#   # THE SPARKS FOUNDATION- Data Science & Business Analytics Internship                            
#                                      Prediction using Supervised Machine Learning
#   
#      In this task it is required to predict the percentage score of a student based on the number of hours studied.
#      The task has to be done by using the Linerar Regression supervised machine learning algorithm.
#      
#   AUTHOR : ANKUSH NEGI
#   
#   STEPS:
#   
#   Step 1 - Importing the dataset
#   
#   Step 2 - Visualizing the Data
#   
#   Step 3 - Data preparation 
#  
#   Step 4 - Training the algorithm
#  
#   Step 5 - Visualising the model
#  
#   Step 6 - Making the prediction
#  
#   Step 7 - Evaluating the model
#   

#   # STEP 1
# 
# Importing the required libraries and the dataset
# 

# In[3]:


# Importing all the required libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# for ignoring the warnings 
import warnings as wg
wg.filterwarnings("ignore")


# In[4]:


# Reading the data from the link

url = "http://bit.ly/w-data"
df=pd.read_csv(url)


# In[7]:


# observing the dataset
df.head()


# In[8]:


df.tail()


# In[9]:


# to find the number of columns and rows of the given dataset
df.shape


# In[10]:


# to find more information about the given dataset
df.info()


# In[11]:


df.describe()


# In[12]:


# now check if the given dataset contains any null or missing values
df.isnull().sum()

So as we can see there are no null or missing values in our dataset so we can move to our next step i.e Visualizing the dataset

# # STEP 2
# 
# Visualizing the Dataset
Here we will plot the dataset to check whether there is any relation betwwen the two variables or not

# In[13]:


# plotting the dataset

plt.rcParams["figure.figsize"] = [15,8]
df.plot(x = 'Hours', y= 'Scores',style= '*', color = 'red', markersize = 10)
plt.title( 'Hours vs percentagee')
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.grid()
plt.show()

From the above graph we can see that there is a relation between 'Hours studied' & 'Percentage score'.
Now we can use this Linear regression supervised model to predict other values.

# In[14]:


# we can also determine the corelation between the variables

df.corr()


# # STEP 3 
# 
# Data Preparation

# In[15]:


# lets take the first 5 data from the dataset

df.head()


# In[16]:


# for dividing the data we will use iloc function

X = df.iloc[:, :1].values
Y = df.iloc[:, 1:].values


# In[17]:


#  Print X
X


# In[18]:


# Print Y
Y


# In[20]:


# now Split the data into training and testing data

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0 )


# # STEP 4
# 
# Training the Algorithm
# 
As we have splitted our data into YTraining & testing sets , now we will train our model
# In[21]:


from sklearn.linear_model import LinearRegression 

model = LinearRegression()
model.fit(X_train, Y_train)


# # STEP 5 
# 
# Visualizing the model
# 
We have train the model so now we will visualise it
# In[22]:


line = model.coef_*X + model.intercept_

# we have to do plotting for te training data


plt.rcParams["figure.figsize"] = [15,8]
df.plot(x = 'Hours', y= 'Scores',style= '*', color = 'red', markersize = 10)
plt.scatter(X_train , Y_train , color='red')
plt.plot(X, line, color='yellow');
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.grid()
plt.show()


# In[23]:


plt.rcParams["figure.figsize"] = [15,8]
plt.scatter(X_test , Y_test , color='red')
plt.plot(X, line, color='yellow');
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.grid()
plt.show()


# # STEP 6
# 
# Making Predictions
We have trained our algorithm , it's time to make some predictions
# In[24]:


#testing data -- in hours
print(X_test) 

# predicting the scores
y_pred = model.predict(X_test)


# In[25]:


# compare actual vs predicted
Y_test


# In[26]:


y_pred


# In[27]:


#comparing actual vs predicted

comp = pd.DataFrame({ 'Actual': [ Y_test],'predicted':[y_pred]})
comp


# In[28]:


# TESTING WITH OUR OWN DATA

hours = 9.5
own_pred = model.predict([[hours]])
print("THE PREDICTED SCORE IF A PERSON STUDIES FOR", hours, "hours is", own_pred[0] )

hence, the predicted score if a person studies for 9.5 hrs is 96.16939661
# # STEP 7
# 
# Evaluating the model
This is the last step , here we are evaluating our trained model by calulating mean absolute error
# In[29]:


from sklearn import metrics

print('MEAN ABSOLUTE ERROR:' , metrics.mean_absolute_error(Y_test, y_pred))

