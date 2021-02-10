# Recommendation Engine Development with IBM Watson Studio Platform

## Executive Summary

In this project, I analyzed the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles I think they will like. 

## Methodology

- Exploratory Data Analysis

- Rank Based Recommendations

- User-User Based Collaborative Filtering

- Matrix Factorization

### Rank Based Recommendations

First, I found the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most
interactions are the most popular. These are then the articles I recommended to new users (or anyone depending on what we know about them).

### User-User Based Collaborative Filtering

I looked at users that are similar in terms of the items they have interacted with and recommended items to similar users.

### Matrix Factorization

I built out a matrix decomposition based on user-item interactions. Using that decomposition, I obtained likelihood of how well I can predict new articles an individual might interact with. It was not as effective as the above two approaches towards this data.
