# Sentiment Analysis Model Comparison

Welcome to the Sentiment Analysis Model Comparison! This document provides an overview of the process involved in training and evaluating various machine learning models for sentiment analysis using restaurant reviews. We will also explore the visualizations of model accuracy and F1-scores.

## Process Overview

1. **Data Loading and Preprocessing**: The dataset of restaurant reviews is loaded, and basic preprocessing steps are applied.

2. **TF-IDF Feature Extraction**: The reviews are transformed into TF-IDF (Term Frequency-Inverse Document Frequency) features, which quantify the importance of words in each review.

3. **Training and Testing Split**: The dataset is divided into training and testing sets, allowing us to train models on one subset and evaluate their performance on another.

4. **Model Selection and Training**: We select several machine learning models for comparison, including Logistic Regression, Naive Bayes, Support Vector Machine (SVM), Gradient Boosting, and k-Nearest Neighbors (KNN). Each model is trained on the training data.

5. **Model Evaluation**: We evaluate the trained models using accuracy, F1-scores, classification reports, and confusion matrices on the testing data.

6. **Model Comparison Visualization**: The accuracy and F1-scores of each model are visualized using bar plots, providing a comprehensive comparison of their performance.

## Models Used

We've chosen a diverse set of models for comparison, each with its strengths and weaknesses:

- **Logistic Regression**: A linear model that's easy to interpret and suitable for binary classification tasks.

- **Naive Bayes**: A probabilistic model that works well with text data and can handle large feature spaces.

- **Support Vector Machine (SVM)**: A powerful algorithm for both linear and non-linear classification tasks.

- **Gradient Boosting**: An ensemble model that combines the predictions of multiple weak learners to create a strong learner.

- **k-Nearest Neighbors (KNN)**: A simple instance-based algorithm that classifies a data point by the majority class among its k-nearest neighbors.

## Graphs and Visualizations

We've created several graphs to help us understand the model performance and comparison:

- **Accuracy Comparison**: We visualize the accuracy of each model using a bar plot, giving us an overall view of their relative performance.

- **F1-Score Comparison**: Additionally, we compare the F1-scores of each model using another bar plot, providing insights into their precision and recall balance.

## Conclusion

This model comparison sheds light on the suitability of different machine learning models for sentiment analysis. By evaluating their accuracy, F1-scores, and reviewing the visualizations, we gain a comprehensive understanding of how these models perform on restaurant review data.

For detailed code and examples, please refer to the Jupyter Notebook provided in this repository.

