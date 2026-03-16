Iris Flower Classification System

📌 Project Overview

The Iris Flower Classification System is a Machine Learning web application that predicts the species of an iris flower based on four input features.

These features describe the physical measurements of the flower:

Sepal Length

Sepal Width

Petal Length

Petal Width

Using these measurements, the system predicts one of the following species:

Iris Setosa

Iris Versicolor

Iris Virginica

This project demonstrates the complete Machine Learning workflow, including data preprocessing, model training, model evaluation, and deployment using a web application.

🎯 Project Objectives

The main goals of this project are:

Understand how Machine Learning classification works.

Train models to classify iris flower species.

Compare the performance of multiple algorithms.

Deploy the trained model using a Flask web application.

Build a simple and interactive user interface for predictions.

🌼 Dataset Information

This project uses the Iris Dataset, one of the most famous datasets in Machine Learning.

Dataset Details


| Feature      | Description                        |
| ------------ | ---------------------------------- |
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width  | Width of the sepal in centimeters  |
| Petal Length | Length of the petal in centimeters |
| Petal Width  | Width of the petal in centimeters  |

Dataset Statistics:

Total Samples: 150

Number of Features: 4

Number of Classes: 3

Class Distribution:

| Species         | Samples |
| --------------- | ------- |
| Iris Setosa     | 50      |
| Iris Versicolor | 50      |
| Iris Virginica  | 50      |

🧠 Machine Learning Models Used

Two classification algorithms were implemented in this project.

1️⃣ K-Nearest Neighbors (KNN)

KNN is a supervised learning algorithm that classifies a data point based on the majority class of its nearest neighbors.

Key Idea:

If a flower is similar to other flowers of a particular species, it will be classified into that species.

Model Performance:

Training Accuracy: 96.67%

Test Accuracy: 100%

2️⃣ Naive Bayes

Naive Bayes is a probability-based classification algorithm based on Bayes' Theorem.

Key Idea:

It calculates the probability that a flower belongs to each class and chooses the most likely one.

Model Performance:

Training Accuracy: 95%

Test Accuracy: 100%

📊 Model Evaluation

To evaluate model performance, the following metrics were used:

Accuracy:

Measures the percentage of correctly predicted samples.

Confusion Matrix:

Shows how many samples were correctly or incorrectly classified.

Classification Report

Includes:

Precision

Recall

F1-Score

These metrics help understand how well the model performs on unseen data.

🖥 Web Application

A Flask web application was developed to allow users to interact with the trained machine learning models.

Features of the Web Interface

User input form for flower measurements

Model selection option (KNN or Naive Bayes)

Predict button to generate results

Display predicted species

Show model performance metrics:

Training Accuracy

Test Accuracy

Confusion Matrix

Classification Report

This makes the system interactive and easy to use.

🛠 Technologies Used
Programming Language

Python

Machine Learning Libraries

Scikit-learn

NumPy

Pandas

Web Framework

Flask

Frontend

HTML

CSS

Model Storage

Pickle (.pkl files)

📈 Example Prediction
Input:

Sepal Length = 5.1

Sepal Width = 3.5

Petal Length = 1.4

Petal Width = 0.2

Output:

Predicted Species:

Iris Setosa

📚 Learning Outcomes

Through this project, the following concepts were learned:

Machine Learning classification

Model training and evaluation

Data preprocessing

Model comparison

Web application deployment using Flask

📚 References

Scikit-learn Documentation

Flask Documentation

UCI Machine Learning Repository (Iris Dataset)

📬 Contact

Name: Neelam. Venkata Chamundeswari 

Mail: chamundeswarineelam10@gmail.com 

Ph.No: 9390530126

Linkidin Profile:https://www.linkedin.com/in/v-chamundeswari-neelam-bb181427b/
