Housing Price Prediction

This project performs data preprocessing, exploratory data analysis (EDA), and machine learning modeling on the California housing dataset. The goal is to predict median house values using multiple approaches, including Perceptron, Least Squares regression, and a Neural Network.

Dataset

The dataset used is housing.csv.zip, which is extracted and loaded into a Pandas DataFrame. It contains various features such as location, population, and income statistics.

Dependencies

Ensure you have the following Python libraries installed:

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras

Steps Performed

1. Data Preprocessing

Extracts and reads the CSV file.

Handles missing values using SimpleImputer (median strategy).

One-hot encodes categorical features (ocean_proximity).

Standardizes numerical features using StandardScaler.

Saves the processed dataset as processed_housing.csv.

2. Exploratory Data Analysis (EDA)

Plots the distribution of house values using a histogram.

Creates a scatter plot of median_income vs. median_house_value.

Generates a heatmap to visualize feature correlations.

3. Machine Learning Models

Perceptron

Implements a simple perceptron classifier to predict whether a house is above or below the median value.

Plots decision boundaries.

Least Squares Regression

Implements a closed-form solution to linear regression.

Computes and evaluates accuracy using sign predictions.

Plots decision boundaries.

Neural Network (Deep Learning)

Constructs a feedforward neural network using Keras.

Trains on 80% of the data and validates on the remaining 20%.

Uses Mean Squared Error (MSE) as the loss function and Adam optimizer.

Visualizes loss over epochs.

Evaluates model performance using MAE.

Compares actual vs. predicted values in a scatter plot.

Running the Code

Execute the Python script to preprocess data, perform EDA, and train models:

python script.py

Results

The perceptron and least squares models provide basic classification and regression insights.

The neural network yields better predictive performance.

Visualizations help analyze data distribution and relationships.

Future Improvements

Use additional feature engineering techniques.

Experiment with different neural network architectures.

Optimize hyperparameters using GridSearch or Bayesian optimization.

Author

This project was developed as part of an exploration in machine learning and deep learning techniques applied to housing price prediction.
