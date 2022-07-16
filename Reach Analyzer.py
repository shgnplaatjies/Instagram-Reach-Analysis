import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor

# To adapt this code: Change the path to the data file, change the figsize values 

def remove_null(data):
    print("Checking for null values in the data")
    if data.isnull().sum().sum() > 0:
        print("Found null values in the data\n",data.isnull().sum()) # Show total  null values 
        data.dropna(inplace=True) # Drop the null values
        print("Dropping null values from the data\n",data.isnull().sum()) # Show total null values
    else:
        print("No null values found in the data")
    print("Checking for data types",data.info()) # Check for data types
    return data

def distImpressionsFromVariousSources(data):
    plt.figure(figsize=(10,8)) # Plot the data with dimensions 10x8
    plt.style.use('fivethirtyeight') # Set the style of the plot
    plt.title("Distribution of Impressions From Home Page") #Set the title of the graph 

    sns.distplot(data['From Home'], label ="From Home") # Plot the distribution of Home impressions in the data 
    sns.distplot(data['From Hashtags'], label ="From Hashtags") 
    sns.distplot(data['From Explore'], label ="From Explore") 

    plt.legend(loc='upper right') # Set the legend location
    plt.show() # Show the distribution of Impressions from Explore Page

# Function to create pie chart of total impressions from various sources
def pieTotalImpressionsVariousSources(data):
    home = data["From Home"].sum() # Calculate the total number of impressions from Home Page, Hashtags, Explore Page, and Other
    hashtags = data["From Hashtags"].sum()
    explore = data["From Explore"].sum()
    other = data["From Other"].sum()

    labels = ['Home', 'Hashtags', 'Explore', 'Other'] # Set the labels for the pie chart
    values = [home, hashtags, explore, other] # Set the values for the pie chart

    fig = px.pie(data, values=values, names=labels, title = 'Impressions on Instagram Posts From Various Sources') # Plot the pie chart
    fig.show()

# Function to plot word cloud of the top 50 most used words in a target feature, returns the word cloud image 
def plot_word_cloud(data, target_feature):
    text = " ".join(data[target_feature]) # Join the captions of the posts into a single string

    stopwords = set(STOPWORDS) # Set the stopwords
    wordcloud = WordCloud(stopwords=stopwords, background_color='white', max_words=50, max_font_size=50).generate(text) # Create the wordcloud

    plt.style.use('classic') # Set the style of the plot
    plt.figure( figsize=(12,10)) # Set the dimensions of the plot
    plt.imshow(wordcloud, interpolation='bilinear') # Plot the wordcloud with bilinear interpolation
    plt.axis("off") # Turn off the axis
    plt.show() # Show the wordcloud
    return wordcloud
    
data = pd.read_csv('Instagram.csv', encoding='latin-1') # Read the data

data = remove_null(data) # Remove null data points

# distImpressionsFromVariousSources(data) # Plot the distribution of Impressions from various sources

# pieTotalImpressionsVariousSources(data) # Plot the pie chart for the total number of impressions from various sources

wordcloud_caption = plot_word_cloud(data, 'Caption') # Plot the wordcloud for the captions of the posts
wordcloud_hashtags = plot_word_cloud(data, 'Hashtags') # Plot the wordcloud for the hashtags of the posts

