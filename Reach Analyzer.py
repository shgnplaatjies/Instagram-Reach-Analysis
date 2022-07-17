import math
from re import T, U
import tarfile
from matplotlib.style import use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import plotly.express as px
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from scipy.stats import linregress

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
def distImpressionsFromVariousSources(data, traffic_features):
    
    plt.figure(figsize=(10,8)) # Plot the data with dimensions 10x8
    plt.style.use('fivethirtyeight') # Set the style of the plot
    plt.title("Distribution of Impressions From Various Sources") #Set the title of the graph 
    for i in traffic_features:
            sns.distplot(data[i], label =i) # Plot the distribution of impressions from all possible various sources in the data                
    plt.legend(loc='upper right') # Set the legend location
    plt.show() # Show the distribution of Impressions from Explore Page

# Function to create pie chart of total impressions from various sources
# Use the "From" keyword to filter the features in the dataset that relate to impressions from a particular source
def pieTotalImpressionsVariousSources(data, traffic_features):
    impression_features_labels = [] # Create a list to store the features that relate to impressions from various sources
    impression_features_values = [] # Create a list to store the summed value of each feature that relates to impressions from various sources
    for i in traffic_features:
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

# Function compares two features based on a given target statistic, leave target as "mean" to compare mean of two features, default is all statistics
# Returns dictionary of features and their respective correlation coefficient
def getStats(data, feature_independent, feature_dependent, target="None"):
    useful_stats_dict = {} # Create a dictionary to store the useful statistics
    useful_stats_dict["correlation"] = data[feature_independent].corr(data[feature_dependent]) # Spearman correlation: Calculate the correlation between the two features
    useful_stats_dict["monotonic correlation"] = data[feature_independent].corr(data[feature_dependent], method='kendall') # Kendall tau correlation: measures the degree of monotonicity of the data.
    useful_stats_dict["covariance"] = data[feature_independent].cov(data[feature_dependent]) # Covariance: Positive if the two variables are positively correlated, negative if they are negatively correlated, and zero if they are uncorrelated
    useful_stats_dict["coefficient of variation"] = data[feature_dependent].std()/data[feature_dependent].mean() # Coefficient of variation: The ratio of the standard deviation to the mean, larger values indicate a higher variance
    useful_stats_dict["variance"] = data[feature_dependent].var() # Variance: The variance of each feature, the higher the value, the more variance there is in the data
    useful_stats_dict["standard deviation"] = data[feature_dependent].std() # Calculate the standard deviation
    useful_stats_dict["mean"] = data[feature_dependent].mean() # Calculate the mean, the average of the feature
    useful_stats_dict["mode"] = data[feature_dependent].mode() # Calculate the mode, the most common value in the data
    useful_stats_dict["median"] = data[feature_dependent].median() # Calculate the median, the middle value of the data
    useful_stats_dict["min"] = data[feature_dependent].min() # Calculate the minimum
    useful_stats_dict["max"] = data[feature_dependent].max() # Calculate the maximum
    useful_stats_dict["skewness"] = data[feature_dependent].skew() # Skewness: The degree of asymmetry of the data, positive values indicate that the most common values (median) are less (to the left) than the average value (mean), negative values indicate that the most common values (median) are more (to the right) than the average value (mean)
    useful_stats_dict["kurtosis"] = data[feature_dependent].kurtosis() # Kurtosis: The degree of peakedness of the data, positive values indicate that the data is peaked, negative values indicate that the data is flat, 0 indicates that the data is symmetric
    useful_stats_dict["5th percentile"] = data[feature_dependent].quantile(0.05) # Calculate the 5th percentile
    useful_stats_dict["25th percentile"] = data[feature_dependent].quantile(0.25) # Calculate the 25th percentile
    useful_stats_dict["50th percentile"] = data[feature_dependent].quantile(0.5) # Calculate the 50th percentile
    useful_stats_dict["75th percentile"] = data[feature_dependent].quantile(0.75) # Calculate the 75th percentile
    useful_stats_dict["95th percentile"] = data[feature_dependent].quantile(0.95) # Calculate the 95th percentile
    useful_stats_dict["99th percentile"] = data[feature_dependent].quantile(0.99) # Calculate the 99th percentile
    useful_stats_dict["sum"] = data[feature_dependent].sum() # Calculate the sum
    useful_stats_dict["count"] = data[feature_dependent].count() # Calculate the count
    useful_stats_dict["range"] = data[feature_dependent].max() - data[feature_dependent].min() # Calculate the range
    useful_stats_dict["count unique"] = data[feature_dependent].nunique() # Calculate the count of unique values
    useful_stats_dict["percent unique"] = data[feature_dependent].nunique()/data[feature_dependent].count() # Calculate the percentage of unique values
    useful_stats_dict["variance unique"] = data[feature_dependent].nunique()/data[feature_dependent].count() # Calculate the variance of unique values
    useful_stats_dict["variance ratio"] = data[feature_dependent].var()/data[feature_dependent].var() # Calculate the variance ratio, between the two features
    
    if target == "None" or target == "all":
        return useful_stats_dict
    elif isinstance(target, list):
            payload = {}
            for i in target:
                payload[i] = useful_stats_dict[i]
            return payload
    return useful_stats_dict.get(target) # Return the useful statistics dictionary
 
  # Function to plot relationship between two features
  # Returns the scatter plot with linear regression line
def scatterPlotWithBestFit(data, feature_independent, feature_dependent):
    
    figure = px.scatter(data_frame=data,x=feature_independent, y=feature_dependent, size =feature_dependent,
               trendline = "ols", title= f"Relationship: Effect of {feature_independent} on {feature_dependent}") # Plot the distribution of impressions from all possible various sources in the data                
    
    return figure

# Function to compare all features based on a given target statistic, leave target as "mean" to compare mean of all features, default is all statistics
def compare_all_features_based_on_target(data, feature_dependent, target="all"):
    for i in data:
        if i != feature_dependent and data[i].dtypes == np.float64:
            getStats(data, feature_dependent, i, target)

# Function to reshape the dataframe to a 1D numpy array
def numpyReshape(data):
    return np.array(data).reshape(-1,1)

# Factory function to create a linear regression model based on 2 given features and a dataframe
# Returns the linear regression model, fit to that data
def linearRegressor(data, feature_independent, feature_dependent):
    linear_regressor = LinearRegression()
    linear_regressor.fit(numpyReshape(data[feature_independent]), numpyReshape(data[feature_dependent])) # Fit the linear regression model to the data
    return linear_regressor

# Function to evaluate the linear regression line between two features
# Returns  average error, r2 score
def evaluateLinearRegression(data, feature_independent, feature_dependent):

    model_likes_from_impressions = linearRegressor(data, 'Impressions', 'Likes') # Fit the linear regression model to the data
    y_Pred = model_likes_from_impressions.predict( numpyReshape(data['Impressions']) ) # Predict the likes based on the Impressions

    average_error = mean_squared_error(numpyReshape(data['Likes']), y_Pred) # Calculate the mean squared error for likes prediction
    r2 = r2_score(numpyReshape(data['Likes']), y_Pred) # Calculate the r2 score for likes prediction, how well the model fits the data
    
    model_evaluation_dict = {   # Create a dictionary to store the model evaluation metrics
                            #  "title": f"Linear Regression: {feature_independent} vs {feature_dependent}",
                             "slope": model_likes_from_impressions.coef_[0],
                             "intercept": model_likes_from_impressions.intercept_,
                             "mean squared error": average_error,
                             "average_error": math.sqrt(average_error), 
                             "r2": r2
                             } 
    
    return model_evaluation_dict

data = pd.read_csv('Instagram.csv', encoding='latin-1') # Read the data
keyword = "From" # Set the keyword (prefix) to filter the features that relate to impressions from a particular source
engagement_features_list = ['Likes', 'Comments', 'Saves', 'Shares', 'Impressions'] # List of features to be used for engagement
traffic_features_list = ['From Home', 'From Hashtags','From Explore', 'From Other'] # List of features to be used for traffic

data = remove_null(data) # Remove null data points

# distImpressionsFromVariousSources(data, traffic_features_list) # Plot the distribution of Impressions from various sources
# pieTotalImpressionsVariousSources(data, traffic_features_list) # Plot the pie chart for the total number of impressions from various sources

# wordcloud_caption = plot_word_cloud(data, 'Caption') # Plot the wordcloud for the captions of the posts
# wordcloud_hashtags = plot_word_cloud(data, 'Hashtags') # Plot the wordcloud for the hashtags of the posts

# compare_all_features_based_on_target(data, 'Impressions', 'unique') # Compare the differences and relationship between Impressions and all other features

metrics_dict = {}

for engagement_feature in engagement_features_list:
    metrics_dict[engagement_feature] = {'title': f'Impressions vs {engagement_feature}'}
    metrics_dict[engagement_feature][f'correlation'] = getStats(data, 'Impressions', engagement_feature, ['correlation', 'mean', 'standard deviation', 'skewness']) # Compare the mean of Impressions and Likes
    metrics_dict[engagement_feature]['model metrics'] = evaluateLinearRegression(data, 'Impressions', engagement_feature) # Evaluate the linear regression model for likes based on Impressions


for metric in metrics_dict:
    print("\n")
    for key in metrics_dict[metric]:
        print(f'{metric}: {key} = {metrics_dict[metric][key]}')
    
    
