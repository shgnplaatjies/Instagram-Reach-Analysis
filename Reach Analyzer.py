import math
import pprint
import tarfile
from tkinter.tix import COLUMN
from unicodedata import numeric
import numpy as np
from re import T, U
import pandas as pd
import seaborn as sns  
import plotly.express as px
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.style import use
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import PassiveAggressiveRegressor
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# To adapt this code: Change the path to the data file, change the figsize values 


# Function to simplify pp.pprint calls
def Print(job):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(job)

# Function to remove null data from the dataframe
def remove_null(data):
    print("Checking for null values in the data")
    if data.isnull().sum().sum() > 0:
        print("Removing null values...")
        # print("Found null values in the data\n",data.isnull().sum()) # Show total  null values 
        data.dropna(inplace=True) # Drop the null values
        # print("Dropping null values from the data\n",data.isnull().sum()) # Show total null values
    else:
        print("No null values found in the data")
    return data

# Function to reshape the dataframe to a 1D numpy array
def numpyReshape(data):
    return np.array(data).reshape(-1,1)

# Factory function create a distribution of impressions from all relevant sources in a given dataset
def distImpressionsFromVariousSources(data, traffic_features, target_feature="all"):
    
    plt.figure() # Plot the data with dimensions 10x8
    plt.style.use('fivethirtyeight') # Set the style of the plot
    plt.title(f"Distribution ofFrom Various Sources") #Set the title of the graph 
    for i in traffic_features:
            sns.distplot(data[i], label =i) # Plot the distribution of impressions from all possible various sources in the data                
    plt.legend(loc='upper right') # Set the legend location
    
    return plt

# Factory function to create pie chart of total impressions from various sources
def pieTotalImpressionsVariousSources(data, target_feature_list):
    pie_chart_labels = [] # Create a list of labels for the pie chart
    pie_chart_values = [] # Create a list of values for the pie chart
    for i in target_feature_list:
            pie_chart_labels.append(i) # Append the feature to the list
            pie_chart_values.append( data[i].sum() ) # Calculate the total number of impressions from all relevant sources

    fig = px.pie(data, values=pie_chart_values, names=pie_chart_labels, title = 'Impressions on Instagram Posts From Various Sources') # Plot the pie chart
    return fig

# Factory function create a word cloud of the top 50 most used words in a target feature, returns the word cloud image 
def createWordCloud(data, target_feature, max_words=100, max_font_size=60, fig_size=(15,15), interpolation='bilinear'):
    string_combined_captions = " ".join(data[target_feature]) # Join the captions of the posts into a single string

    stopwords = set(STOPWORDS) # Set the stopwords to default
    target_wordcloud = WordCloud(stopwords=stopwords, background_color='white', max_words=max_words, max_font_size=max_font_size)
    target_wordcloud.generate(string_combined_captions) # Create the WordCloud image from the combined captions
    print(target_wordcloud.fit_words) # Print the top 50 most used words in the captions
    target_wordcloud.to_file(f'Wordcloud for {target_feature} - {max_words} words, font size {max_font_size} .png') # Save the WordCloud image to a file
    plt.style.use('classic') # Set the style of the image
    plt.figure( figsize=fig_size) # Set the dimensions of the image
    plt.imshow(target_wordcloud, interpolation=interpolation) # Plot the WordCloud, using bilinear interpolation
    plt.axis("off") # Turn off the axis
    
    return plt

# Function compares two features based on a given target statistic, leave target as "mean" to compare mean of two features, default is all statistics
def getStats(data, feature_independent, feature_dependent, target="all"):
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
    useful_stats_dict["skewness"] = data[feature_dependent].skew() # Skewness: The degree of asymmetry of the data, positive values indicate most of the datapoints are above average, negative values indicate most of the datapoints are below average.
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
    
    obj2 = evaluateLinearRegression(data, feature_independent, feature_dependent) # Evaluate the linear regression model for likes based on Impressions
    for i in obj2:
        useful_stats_dict[i] = obj2[i]

    if target == "all":
        return useful_stats_dict
    elif isinstance(target, list):
            payload = {}
            for i in target:
                payload[i] = useful_stats_dict[i]
            return payload
    return useful_stats_dict.get(target) # Return the useful statistics dictionary
 
  # Function to plot relationship between two features, with regression line
def scatterPlotWithBestFit(data, feature_independent, feature_dependent):
    
    figure = px.scatter(data_frame=data,x=feature_independent, y=feature_dependent, size =feature_dependent,
               trendline = "ols", title= f"Relationship: Effect of {feature_independent} on {feature_dependent}") # Plot the distribution of impressions from all possible various sources in the data                
    
    return figure

# Function to compare all features based on a given target statistic, leave target as "mean" to compare mean of all features, default is all statistics, features list "None to compare 1 feature to itself"
def compareListOfFeaturesToFeature(data, feature_independent, features_list="None", target="all"):
    stats_dict = {}
    if features_list == "None":
        features_list = data.columns.to_list()
    for feature in data[features_list]:
        if feature != feature_independent and data[feature].dtypes == np.float64:
            stats_dict[f'{feature}'] = getStats(data, feature_independent, feature, target)

    return stats_dict

# Factory function to create a linear regression model based on 2 given features and a dataframe
def linearRegressor(data, feature_independent, feature_dependent):
    linear_regressor = LinearRegression()
    linear_regressor.fit(numpyReshape(data[feature_independent]), numpyReshape(data[feature_dependent])) # Fit the linear regression model to the data
    return linear_regressor

# Function to evaluate the linear regression line between two features
def evaluateLinearRegression(data, feature_independent, feature_dependent):

    modelLinear = linearRegressor(data, feature_independent, feature_dependent) # Fit the linear regression model to the data
    y_Pred = modelLinear.predict( numpyReshape(data[feature_independent]) ) # Predict the likes based on the Impressions

    meanSquaredError = mean_squared_error(numpyReshape(data[feature_dependent]), y_Pred) # Calculate the mean squared error for likes prediction
    r_squared = r2_score(numpyReshape(data[feature_dependent]), y_Pred) # Calculate the r2 score for likes prediction, how well the model fits the data
    
    model_evaluation_dict = {   # Create a dictionary to store the model evaluation metrics
                            #  "title": f"Linear Regression: {feature_independent} vs {feature_dependent}",
                             "slope": modelLinear.coef_[0],
                             "intercept": modelLinear.intercept_,
                             "mean squared error": meanSquaredError,
                             "average_error": math.sqrt(meanSquaredError), 
                             "r2": r_squared
                             } 
    
    return model_evaluation_dict

# Function returns sorted dictionary by key, descending order
def sortDictDescending(dict):
    sorted_tuples = sorted(dict.items(), key=lambda item: item[1])
    print(sorted_tuples)
    sorted_dict = {k: v for k, v in sorted_tuples}
    return sorted_dict

# Function returns sorted dictionary by key, ascending order
def sortDictAscending(dict):
    return sorted(dict.items(), key=lambda x: x[1])

# Function to get metrics of 2 features based on a given target statistic, default is all statistics
def getStatsRelativeToTarget(data, relevant_features, feature_independent, target_stats="all"):
    metrics_dict = {}

    for feature in relevant_features: # Iterate through relevant features
        # scatterPlotWithBestFit(data, feature_independent, feature).show() # Plot the relationship between Impressions and the given feature
        metrics_dict[feature] = {'independent variable': f'{feature_independent}', "dependent variable": f'{feature}'}
        obj1 = getStats(data, feature_independent, feature, target_stats) # Compare the mean of Impressions and Likes
        obj2 = evaluateLinearRegression(data, feature_independent, feature) # Evaluate the linear regression model for likes based on Impressions
        for i in obj1:
            metrics_dict[feature][i] = obj1[i]
        for i in obj2:
            metrics_dict[feature][i] = obj2[i]

    sorted_data = pd.DataFrame.from_dict(metrics_dict, orient='index')
    return sorted_data

# Functoin to sort a given dataframe by a given column, in descending order
def sortDataframeDescending(data, target):
    return data.sort_values(by=target, ascending=False)

# Functoin to sort a given dataframe by a given column, in descending order
def sortDataframeAscending(data, target):
    return data.sort_values(by=target, ascending=True)

# Functoion to create a CSV file from a given dataframe, with a given filename
def createCSV(data, target_feature='independent variable'):
    title = data_r2.get(target_feature)[0]
    if target_feature == "independent variable" and title != "None":
        filename = f"Metrics related to {title}.csv"
        data.to_csv(filename, index=False)
        print(f"Created \"Metrics related to {title}.csv\"")
    elif title == "None":
        filename = f"Metrics related to {title} - check for corrupt data.csv"
        data.to_csv(f"{filename}.csv", index=False)
        print(f"Created {filename}")
    else:
        filename = f"Target_feature metrics related to {title}.csv"
        data.to_csv(f"{filename}.csv", index=False)
        print(f"Created {filename}")

# Get the conversion rate trend for a given feature, default is Follows but could also make sense for purchases or a target activity like Shares/Saves
def getFeatureConversionData(data, feature_independent, feature_dependent = 'Follows'): 
    conversion_rate,x,y = [],[],[]
    
    for i in data[feature_dependent]:
        x.append(i)
    for i in data[feature_independent]:
        y.append(i)
    for i in range(len(x)):
        if y[i]!= 0:
            conversion_rate.append((x[i]/y[i]))
    
    return conversion_rate

# Function to get the total conversion rate across the sum of given feature, default is Follows but could also make sense for purchases or a target activity like Shares/Saves
def getFeatureConversionRateTotal(data, feature_independent, feature_dependent = 'Follows'): 
    return data[feature_dependent].sum() / data[feature_independent].sum() # Calculate the total conversion rate for the given feature
 
# Factory function create a the conversion rate of a target feature (Follows or Sales) against a given feature, default is Profile Visits. This is useful for finding the best feature to optimize for the conversion rate
def plotConversionRate(data, conversion_feature = 'Profile Visits', conversion_target = 'Follows'):
    conversion_rate = getFeatureConversionData(data, conversion_feature) # Get the follower conversion rate for feature, Set the feature to be analyzed for conversion against the target feature (usually 'Follows' or 'Sales')
    plot = px.scatter(data, y=conversion_rate, x=data[conversion_feature], size =conversion_feature,
                      trendline = "ols", title=f"Relationship: Effect of {conversion_feature} on {conversion_target} conversion rate") # Plot the distribution of impressions from all possible various sources in the data
    return plot
   
# Factory function to create a passive aggressive regressor (linear regression) model to predict a target, for a given set of features
def getModelPassiveAggressiveRegressor(data, features_list, target):
    features = np.array(data[features_list])
    y = np.array(data[target])
    
    x_train, x_validation, y_train, y_validation = train_test_split(features, y, test_size=0.2, random_state=42)
    
    model = PassiveAggressiveRegressor()
    model.fit(x_train, y_train)
    score = model.score(x_validation, y_validation)
    print("Model Accuracy on Validation Data:",score)
    
    return model

# Function returns a datapoint for the specified set of features and at their specified target values. i.e. test maximum data point for the given features, test average data point for the given features, test minimum data point for the given features 
def getTargetStub(data,y, features_list,  target='max'):
    test = compareListOfFeaturesToFeature(data, y, features_list, target=[target])
    test_stub = []
    for i in test.values():
        test_stub.append(i[target])
    return test_stub

# Function to plot scatter plot of a given feature against a given target, default is all features and scatter 
def plotFeatureVsTarget(data, target, feature, color = "", size="", type="scatter", show="False"):
    if color == "": color = feature
    if size == "": size = feature
    
    match type:
        case "scatter":
            plot = px.scatter(data, y=data[feature], x=data[target], size =data[size], color=data[color],  trendline = "ols", title=f"{type} plot: Effect of {feature} on {target}")
    
        case "scatter iris":
            new_data = px.data.iris()
            plot = px.scatter(new_data, y=data[feature], x=data[target], size =data[size], trendline = "ols", title=f"{type} plot: Effect of {feature} on {target}")
    
        case "area":
            plot = px.area(data, y=data[feature], x=data[target], color=data[color], title=f"{type} plot: Effect of {feature} on {target}")
    
        case "line":
            plot = px.line(data, y=data[feature], x=data[target], color =target, title=f"{type} plot: Effect of {feature} on {target}")
        
        case "bar":
            plot = px.bar(data, y=data[feature], x=data[target], title=f"{type} plot: Effect of {feature} on {target}")
        
        case "scatter_3d":
            plot = px.scatter_3d(data, y=data[feature], x=data[target], size =feature, title=f"{type} plot: Effect of {feature} on {target}")
        
        case "line_3d":
            plot = px.line_3d(data, y=data[feature], x=data[target], title=f"{type} plot: Effect of {feature} on {target}")
            
        case "box": 
            plot = px.box(data, y=data[feature], x=data[target], title=f"{type} plot: Effect of {feature} on {target}")
        
        case "violin":
            plot = px.violin(data, y=data[feature], x=data[target], title=f"{type} plot: Effect of {feature} on {target}")
        
        case "histogram":
            plot = px.histogram(data, y=data[feature], x=data[target], title=f"{type} plot: Effect of {feature} on {target}")
        
        case "distplot":
            plot = px.distplot(data, y=data[feature], x=data[target], title=f"{type} plot: Effect of {feature} on {target}")
        
        case "scatter_matrix":
            new_data = toIris(data) # Convert the dataframe to the Iris dataset format
            plot = px.scatter_matrix(new_data)
        case default:
            plot = px.scatter(data, y=data[feature], x=data[target], size =feature, trendline = "ols", title=f"{type} plot: Effect of {feature} on {target}")
    if show == "True":
        plot.show()
    return plot

def toIris(data):
    new_data = px.data.iris()
    return new_data

def main():
    data = pd.read_csv('Instagram.csv', encoding='latin-1') # Read the data
    data = remove_null(data) # Remove null data points
    
    # Target stats to use to test the stats evaluations methods
    target_stats = ['correlation', 
                    'r2', 
                    'coefficient of variation',
                    'mean',
                    'standard deviation',
                    'skewness'
                    ]

    # Features in each category, exact categories and their content are specific to this data set.

    engagement_features_list = ['Likes', # List of features to be used for engagement
                                'Comments', 
                                'Shares', 
                                'Saves'] 
    
    impressions_features_list = ['From Home', # List of features to be used for traffic
                                 'From Hashtags',
                                 'From Explore', 
                                 'From Other'
                                 ] 
    
    numerical_features_list = ['Impressions', # List of features to be used for numerical values
                               'From Home', 
                               'From Hashtags',
                               'From Explore', 
                               'From Other', 
                               'Likes', 
                               'Comments', 
                               'Shares', 
                               'Saves', 
                               'Profile Visits', 
                               'Follows'
                               ] 
    
    text_features_list = ['Hashtags', # List of features to be used for text data
                          'Caption'
                          ] 
    
    
   
    # The name of each plot category and the plot types that are available for them.
    #______________________________________________________________________________________________________________________________
    
    basic = ['scatter', # Basic type of plot
             'line',
             'area', 
             'bar', 
             'funnel', 
             'timeline']
  
    part_of_whole = ['pie', # Types of plot that show how certain data fits as part of a greater 'whole' data set
                     'sunburst',
                     'treemap',
                     'icile',
                     'funnel_area']
   
    distribution_1d = ['histogram',  # Types of plot that show 1D distributions of data
                       'distplot',
                       'box', 
                       'violin', 
                       'strip', 
                       'ecdf']
  
    distribution_2d = ['density', # Types of plot that show 2D distributions of data, 2 axis so 2 features (columns)
                       'heatmap', 
                       'density_contour']
  
    distribution_3d = ['scatter_3d', # Types of plot that show 3D distributions of data, 3 axis so 3 features (columns)
                       'line_3d']
  
    matrix = ['scatter_matrix', # Types of plot that show a matrix of data, (vector of columns)
              'imshow']
  
    multi_dimensional = ['scatter_matrix', # Types of plot that show a multi dimensional data
                         'imshow',
                         'parallel_coordinates',
                         'parallel_categories']
  
    tile_maps = ['scatter_mapbox', # Types of plot that show a tile map of the data
                 'line_mapbox', 
                 'chloropleth_mapbox',
                 'density_mapbox']
  
    outline_maps = ['scatter_geo', # Types of plot that show a outline map of the data
                    'line_geo',
                    'chloropleth']
   
    polar_charts = [ 'scatter_polar', # Types of plot that show a polar chart of the data
                    'line_polar',
                    'bar_polar']
    
    ternary_charts = ['ternary'] # Types of plot that show a ternary chart of the data
    
    text_charts = ['text'] # Types of plot that show a text charts and word clouds of the data, categorical input
    
    
     # List of all available plot categories on plotly
    #______________________________________________________________________________________________________________________________
    
    plot_categories_dict = {"basic" : basic, 
                             "part_of_whole": part_of_whole,
                             "distribution_1d" : distribution_1d, 
                             "distribution_2d" : distribution_2d,
                             "distribution_3d"  : distribution_3d,
                             "matrix" : matrix,
                             "image" : matrix,
                             "multi_dimensional" : multi_dimensional,
                             "tile_maps" : tile_maps,
                             "outline_maps" : outline_maps,
                             "polar_charts" : polar_charts,
                             "ternary_charts" : ternary_charts,
                             "text_charts" : text_charts
                             } 
    

    
    
    
    #______________________________________________________________________________________________________________________________

    # Plotting the data using the inbuilt functions
    #______________________________________________________________________________________________________________________________
    
    # norm_distr_impressions_all_sources =  distImpressionsFromVariousSources(data, impressions_features_list) # Plot the distribution of Impressions from various sources
    # norm_distr_impressions_all_sources.show()

    # pie_chart = pieTotalImpressionsVariousSources(data, traffic_features_list) # Plot the pie chart for the total number of impressions from various sources
    # pie_chart.show()

    # wordcloud_captions = createWordCloud(data, 'Caption') # Plot the wordcloud for the captions of the posts
    # wordcloud_captions.show() # Plot the wordcloud for the captions of the posts
    # wordcloud_hashtags = createWordCloud(data, 'Hashtags') # Plot the wordcloud for the hashtags of the posts
    # wordcloud_hashtags.show() # Plot the wordcloud for the hashtags of the posts

    # feature_independent = 'Likes' # Set the feature to be analyzed
    # stats_for_data = compareListOfFeaturesToFeature(data, feature_independent, features_list= ['From Explore', 'Follows', 'Profile Visits'], target=target_stats) # Compare the differences and relationship between Impressions and all other features
    # Print([f"Getting stats for {feature_independent}",stats_for_data])

    # target_stats = ['correlation', 'coefficient of variation', 'mean', 'standard deviation', 'skewness']
    # metricsImpressionsAll = getStatsRelativeToTarget(data, engagement_features_list, 'Impressions') # Get the metrics for Impressions
    # print(f"All metrics related to {'Impressions'}", metricsImpressionsAll)

    # data_r2 = sortDataframeDescending(metricsImpressionsAll, 'r2') # Sort the dataframe by r2 score, more accurate than the correlation value
    # createCSV(data_r2) # Create a CSV file for the metrics
    
    # conversion_plot = plotConversionRate(data, 'Profile Visits') # Plot the conversion rate for the given feature
    # conversion_plot.show() # Plot the conversion rate for the given feature

    # total_conversion_rate = getFeatureConversionRateTotal(data, 'Profile Visits') # Get the total conversion rate for the given feature
    # Print([f"Total conversion rate of {'Profile Visits'}",total_conversion_rate])
    
    #______________________________________________________________________________________________________________________________
    
    features_list = ['Likes','Saves','Comments','Shares','Profile Visits', 'Follows']
    y = 'Impressions'
    test_stub_for_features = np.array([getTargetStub(data, y, features_list, 'max')])
    model = getModelPassiveAggressiveRegressor(data, features_list, y)

    
    y_feature = "Likes" 
    x_feature = "Impressions"
    
    colors = ['Likes', 'Comments', 'Shares', 'Saves']
    sizes = ['From Home', 'From Hashtags','From Explore', 'From Other']
    
    # Testing the plotting categories, trying to make it easy to visualize different types of data, while being able to configure color, size, overlay multiple distributions on one another etc. 
    for plot_type in plot_categories_dict["basic"]:
        plot = plotFeatureVsTarget(data, x_feature, y_feature, type = plot_type)
        plot.show()
        plot = plotFeatureVsTarget(data, x_feature, y_feature, type = plot_type, size = x_feature)
        plot.show()
        plot = plotFeatureVsTarget(data, x_feature, y_feature, type = plot_type, color = x_feature)
        plot.show()
        plot = plotFeatureVsTarget(data, x_feature, y_feature, type = plot_type, color = x_feature, size = x_feature)
        plot.show()

    y_prediction = model.predict(test_stub_for_features)
    print(y_prediction)

if __name__ == "__main__":
    main()


