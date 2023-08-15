# Exploratory Data Analysis of Restaurant Reviews

Welcome to the Exploratory Data Analysis (EDA) of restaurant reviews! This document provides a comprehensive analysis of restaurant reviews using Python and data visualization techniques.

## Table of Contents

1. [Data Loading and Basic Exploration](#data-loading-and-basic-exploration)
2. [Review Length Analysis](#review-length-analysis)
3. [Common Words Analysis](#common-words-analysis)
4. [Sentiment-based Comparison](#sentiment-based-comparison)
5. [N-Grams Analysis](#n-grams-analysis)
6. [Conclusion](#conclusion)

## Data Loading and Basic Exploration

In this initial step, the dataset is loaded from a TSV file using Pandas, and basic exploratory insights are gained.

## Review Length Analysis

Analyzing the distribution and lengths of reviews is crucial to understand the overall spread of review text.

### - Distribution of Review Lengths
Visualizing the distribution of review lengths using a histogram helps us identify the most common lengths and any potential outliers.

## Common Words Analysis

Understanding the most common words used in both positive and negative reviews provides insights into the overall sentiment and themes expressed by reviewers.

### - Most Common Words in Positive Reviews
Identifying the most common words in positive reviews sheds light on the positive aspects highlighted by customers.

### - Most Common Words in Negative Reviews
Exploring the most common words in negative reviews reveals the pain points and concerns voiced by customers.

## Sentiment-based Comparison

Comparing positive and negative reviews enables us to understand the differences in review lengths and frequently used words between the two sentiments.

### - Positive Review Word Cloud
A word cloud for positive reviews visually represents frequently occurring words, giving a quick overview of positive sentiments.

### - Negative Review Word Cloud
Similarly, a word cloud for negative reviews visually represents frequently used words, providing insight into negative sentiments.

### - Positive and Negative Box Plots of Review Lengths
Comparing box plots of review lengths between positive and negative reviews highlights differences in review lengths.

### - Positive and Negative Reviews Lengths by Sentiment
Using kernel density estimates, we compare the distribution of review lengths by sentiment, revealing patterns in review length variation.

## N-Grams Analysis

N-grams (combinations of adjacent words) offer deeper insights into common phrases and expressions used in reviews.

### - Top N-Grams in Positive Reviews
Analyzing top N-grams in positive reviews helps us identify commonly expressed positive sentiments and experiences.

### - Top N-Grams in Negative Reviews
Analyzing top N-grams in negative reviews highlights commonly shared negative experiences and concerns.

## Conclusion

This EDA provides valuable insights into review length distribution, common words in positive and negative sentiments, and frequently used phrases. Utilize these insights to inform further analysis and decision-making.

For detailed code and examples, please refer to the Jupyter Notebook provided in this repository.
