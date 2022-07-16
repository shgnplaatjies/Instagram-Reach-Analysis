import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv('Instagram.csv', encoding='latin-1') # Read the data
print("Checking for null values in the data")
if data.isnull().sum().sum() > 0:
    print("Found null values in the data\n",data.isnull().sum()) # Check for null values 
    data.dropna(inplace=True) # Drop the null values
    print("Dropping null values from the data\n",data.isnull().sum()) # Check for null values
else:
    print("No null values found in the data")

print("Checking for data types",data.info()) # Check for data types

plt.figure(figsize=(10,8)) # Plot the data with dimensions 10x8
plt.style.use('fivethirtyeight') # Set the style of the plot
plt.title("Distribution of Impressions From Home Page") #Set the title of the graph 

sns.distplot(data['From Home'], label ="From Home") # Plot the distribution of Home impressions in the data | Blue 
sns.distplot(data['From Hashtags'], label ="From Hashtags") # Plot the distribution of Hashtags impressions in the data | Red
sns.distplot(data['From Explore'], label ="From Explore") # Plot the distribution of Explore impressions in the data | Yellow

plt.legend(loc='upper right') # Set the legend location
plt.show() # Show the distribution of Impressions from Explore Page




