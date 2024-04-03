#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go


# In[2]:


df=pd.read_csv(r"C:\Users\Sakib\Downloads\Bangladessh.project1\road_accident_statistics.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.info()


# In[8]:


# Convert 'Year' column to its original format
df['Year'] = pd.to_numeric(df['Year'])

# Now you can display the summary statistics
print(df.describe())


# In[9]:


df.isnull().sum()


# In[10]:


df.Year.describe()


# In[11]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Year', bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.show()


# In[12]:


df.head()


# In[ ]:





# In[13]:


for column in df.columns[1:]:  # Exclude 'Year' column
    plt.figure(figsize=(5,4))
    sns.histplot(data=df, x=column, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# 

# In[14]:


variables = ['Death', 'Number of Serious Injuries', 'Number of Minor Injuries', 
             'Number of Moderate Injuries', 'Number of Severe Injuries']

for var in variables:
    # Scatter Plot
    plt.figure(figsize=(12, 4))
    sns.scatterplot(data=df, x='Year', y=var)
    plt.title(f'Scatter Plot: Year vs. {var}')
    plt.xlabel('Year')
    plt.ylabel(var)
    plt.show()

    # Line Plot
    plt.figure(figsize=(12, 4))
    sns.lineplot(data=df, x='Year', y=var)
    plt.title(f'Line Plot: Year vs. {var}')
    plt.xlabel('Year')
    plt.ylabel(var)
    plt.show()

    # Box Plot
    plt.figure(figsize=(25, 12))
    sns.boxplot(data=df, x='Year', y=var)
    plt.title(f'Box Plot: Year vs. {var}')
    plt.xlabel('Year')
    plt.ylabel(var)
    plt.show()


# In[16]:


# Pair Plot
plt.figure(figsize=(12, 10))
sns.pairplot(df)
plt.suptitle('Pair Plot', y=1.02)
plt.show()




# In[17]:


# Correlation Matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[19]:


df.columns


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Features
X = df[['Year']]

# Target Variables
y_columns = ['Number of Accidents', 'Death', 'Number of Serious Injuries',
             'Number of Minor Injuries', 'Number of Moderate Injuries',
             'Number of Severe Injuries']

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, df[y_columns], test_size=0.2, random_state=42)

# Models
models = {
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Linear Regression": LinearRegression()
}

# Model Training and Evaluation
for target_column in y_columns:
    print(f"\nPredicting for target: {target_column}\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train[target_column])
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test[target_column], y_pred)
        print(f"Mean Absolute Error for {name}: {mae}")


# In[23]:


# New data point for the year 2025
new_year = [[2025]]

# Predictions for the year 2025 using each model
for target_column in y_columns:
    print(f"\nPredictions for target: {target_column}\n")
    for name, model in models.items():
        # Predict using the model
        prediction = model.predict(new_year)
        print(f"Predicted {target_column} ({name}): {prediction}")


# In[24]:


# Dictionary to store mean absolute errors for each model
mae_scores = {}

# Model Training and Evaluation
for target_column in y_columns:
    print(f"\nPredicting for target: {target_column}\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train[target_column])
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test[target_column], y_pred)
        mae_scores[f"{name} ({target_column})"] = mae
        print(f"Mean Absolute Error for {name}: {mae}")

# Find the best-performing model based on the lowest MAE
best_model = min(mae_scores, key=mae_scores.get)
print(f"\nBest Model: {best_model} with MAE of {mae_scores[best_model]}")


# In[ ]:




