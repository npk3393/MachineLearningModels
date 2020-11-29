# Data Scientist Capstone Project

## Sparkly

This capstone project is part of requirements of the Udacity data scientist nano degree

In this project, I utilized thousands of JSON documents (log files) of an online music streaming service, called Sparkify to understand the behaviors and patterns of the listeners.

## Executive Summary

The project starts off with importing the dataset , performing univariate and bivariate analysis. It then shapes into transforming the data set for further analysis and modeling, removing missing values and deduplications, and eventually creating features for machine learning modeling. Machine learning models have been employed on this prepared data set to understand and predict the churning patterns of subscribers.

I have utilized F1 score and accuracy metric to understand the model and evaluate the best fit. A blog post has also been written to supplement the project. Logistic Regression performed better after comparing with decision tree classifier, yielded better prediction results.

## Technical Environment

The project uses apache spark that sits on top of apache Hadoop. Data is stored in Amazon S3 that is being called from the script. We use PySpark to perform analysis and analysis an API that is available to us to utilize spark functions.

## Python Packages

PySpark, Numpy, Pandas, Matplotlib, DateTime

## Models

Logistic Regression, Decision Tree Classifier 

## Methodology

The notebook contains executable scripts and transferable python code that performs:

Exploratory Data Analysis

Univariate and Bi-variate Data Visualizations

Data Cleaning and Transformation

Feature Engineering for Predictive Modeling

Machine Learning Modeling

Conclusion Remarks

## Summary of Results

We used an initial base modeling and then implemented pipelines and hyperparameter tuning for both logit and decision tree models.
Both yielded the following scores:

The base logit model has an accuracy of 0.7846153846153846 and F1-score of 0.7659763313609467 where as the base decision tree model has an accuracy of 0.8307692307692308 and F1-score of 0.8324610097805973. 

My initial reaction was, this was probably due to model over fitting as no hyperparameter were tuned.

I tuned the regularization parameter for the logit model and maxDepth of trees for decision tree parameter and this yielded the following results.

The tuned logit model has an accuracy of ... and F1-score of ... where as the tuned decision tree model has an accuracy of ... and F1-score of ...

I attribute these metrics to the quality of features being engineered and thorough data cleansing process

## File Descriptions

The folder contains IPython notebooks Sparkify-Copy1.ipynb and Sparkify-Copy2.ipynb are 1st and 2nd versions respectively

## Necessary Acknowledgements

I would like to acknowledge Stackoverflow, Geeks for Geeks, Python documentation and various other google search results to help me understand various functions, and options exist within PySpark 


## Remarks

The code uses both Spark Data Frames and Spark SQL for exploratory data analysis and other purposes.
Here is the blog link: https://pavan-narayanan.medium.com/predicting-customer-churning-using-machine-learning-bfb8883f46c0

