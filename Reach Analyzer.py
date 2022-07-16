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

# Function to plot distribution of impressions from all relevant sources in a given dataset
# Use the "From" keyword to filter the features in the dataset that relate to impressions from a particular source
def distImpressionsFromVariousSources(data, keyword):
    
    plt.figure(figsize=(10,8)) # Plot the data with dimensions 10x8
    plt.style.use('fivethirtyeight') # Set the style of the plot
    plt.title("Distribution of Impressions From Various Sources") #Set the title of the graph 
    for i in data:
            if str(i)[:3] == keyword[:3]: # Check if the feature starts with "From" and if it does, plot the data. That means that impressions can come from this feature
                sns.distplot(data[i], label =i) # Plot the distribution of impressions from all possible various sources in the data                
    plt.legend(loc='upper right') # Set the legend location
    plt.show() # Show the distribution of Impressions from Explore Page

# Function to create pie chart of total impressions from various sources
# Use the "From" keyword to filter the features in the dataset that relate to impressions from a particular source
def pieTotalImpressionsVariousSources(data, keyword):
    impression_features_labels = [] # Create a list to store the features that relate to impressions from various sources
    impression_features_values = [] # Create a list to store the summed value of each feature that relates to impressions from various sources
    for i in data:
            if str(i)[:3] == keyword[:3]: # Check if the feature starts with "From" and if it does, plot the data. (verbose) That means that impressions can come from this feature
                impression_features_labels.append(i) # Append the feature to the list
                impression_features_values.append( data[i].sum() ) # Calculate the total number of impressions from all relevant sources

    print(impression_features_labels, impression_features_values)
    labels = impression_features_labels # Set the labels for the pie chart
    values = impression_features_values # Set the values for the pie chart

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
keyword = "From" # Set the keyword (prefix) to filter the features that relate to impressions from a particular source

data = remove_null(data) # Remove null data points

distImpressionsFromVariousSources(data, keyword) # Plot the distribution of Impressions from various sources
pieTotalImpressionsVariousSources(data, keyword) # Plot the pie chart for the total number of impressions from various sources

# wordcloud_caption = plot_word_cloud(data, 'Caption') # Plot the wordcloud for the captions of the posts
# wordcloud_hashtags = plot_word_cloud(data, 'Hashtags') # Plot the wordcloud for the hashtags of the posts


