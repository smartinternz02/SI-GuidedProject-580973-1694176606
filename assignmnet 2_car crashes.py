#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seaborn')


# In[2]:


import seaborn as sns


# In[3]:


print(sns.get_dataset_names())


# In[4]:


df=sns.load_dataset('car_crashes')
#loading dataset car_crashes from seaborn library


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.head()


# In[11]:


sns.scatterplot(x="total", y="speeding", data=df)

# Inference:
# Relation between total accidents and ones involved in speeding
# From below scatter plot graph -- as speed increases, total car crashes also increases


# In[12]:


sns.scatterplot(x="alcohol",y="total",data=df)

# Relation between total accidents and ones involved in alcohol drinking
# As alcohol drinking accidents increases total accidents also increases


# In[13]:


sns.scatterplot(x="ins_losses",y="alcohol",data=df)

# There is no certain relation between ins_losses and alochol sometimes it works with direct propostionality and sometimes with inverse proportionality


# In[14]:


import matplotlib.pyplot as plt

sns.scatterplot(data=df, x='alcohol', y='speeding')
plt.xlabel('Percentage of Drivers Involved in Alcohol Accidents')
plt.ylabel('Number of Speeding Accidents')
plt.title('Relationship between Alcohol and Speeding Accidents')
plt.show()

# Scatter plot graph between alcohol and speeding accidents and labelled x and y axis separately as 'Percentage of Drivers Involved in Alcohol Accidents' and 'Number of Speeding Accidents'
# If Percentage of Drivers Involved in Alcohol Accidents increases then Number of Speeding Accidents also increases


# In[15]:


sns.lineplot(x="alcohol", y="total", data=df)

# transparent area - confidence interval(shows range)
# lineplot between total accidents and Drivers Involved in Alcohol Accidents


# In[16]:


sns.lineplot(x="speeding", y="total", data=df)

# lineplot between total accidents and speeding Accidents


# In[31]:


sns.distplot(df["alcohol"])

# Distplot is a univariate distribution of observations
# It combines matplotlib histogram and kernel distribution plot
# univariate distribution of accidents invloved in alcohol driving in car_crashes dataset
# involved in alcohol incident ranges from 4-6 %


# In[32]:


sns.distplot(df["ins_premium"])

# insurance premium ranges from 800 - 1000 dollars


# In[20]:


sns.relplot(x="total",y="speeding",data=df,hue="abbrev")

# It allows us to visualise how variables within a dataset relate to each other
# hue paramater  - says which column in dataframe is need for color encoding (categorical variable)
# scatter plot between total accidents and speeding relating to abbrev column


# In[21]:


df["abbrev"].value_counts()
# checking total count of 'abbrev' feature which is a multivariate


# In[22]:


sns.barplot(data=df,x="abbrev",y="speeding", ci=None)

# graph b/w categorical and numerical variable
# Bar graph plot between abbrev and speeding
# can dtect which abbrev has high speed in accidents


# In[23]:


sns.barplot(data=df,x="abbrev",y="speeding", hue="abbrev")   #color coding - hue

# Bar graph plot between categorical and numerical variable i.e abbrev and speeding


# In[24]:


sns.countplot(x="abbrev",data=df)

# Used to represent the occurrence(counts) of the observation present in the categorical variable.


# In[25]:


sns.jointplot(x="total",y="ins_losses",data=df)

# dispalys relationship b/w total total accidents and insurance losses
# Relationship between two variables and the distribution of individuals of each variable.
# It  is bivariate analysis -  2 var at a time


# In[26]:


sns.jointplot(x="total",y="abbrev",data=df)

# dispalys relationship b/w total total accidents and abbrev variables
# Relationship between two variables and the distribution of individuals of each variable which is  a bivariate analysis


# In[27]:


sns.boxplot(x="total",y="alcohol", data=df)

# Compares the interquartile ranges (that is, the box lengths) to examine how the data is dispersed between each sample


# In[29]:


corr=df.corr()
corr

# correlation --- statistical analysis relation b/w 2 variables(either directly prop or inversely prop)
# values > 0.5 --- highly correlated
# values < 0.5 --- less correlated


# In[45]:


sns.heatmap(corr,annot=True)

# darker shade - less correlation
# lighter shade - highly correlated
# It represents how each feature depends on the other features


# In[30]:




