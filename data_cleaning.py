# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# File paths
train_file_path = 'train.csv'
test_file_path = 'test.csv'
cleaned_train_file_path = 'cleaned_train_data.csv'
cleaned_test_file_path = 'cleaned_test_data.csv'

# Loading and reading the data (Train & Test)
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Display summary statistics for numerical values
print(train_data.describe())

# Checking for missing values
print("Missing values in train data:\n", train_data.isnull().sum())
print("Missing values in test data:\n", test_data.isnull().sum())

# Visualizing outliers using boxplots before cleaning
def plot_boxplot(df, column, title):
    """
    Plots a boxplot for the specified column in the dataframe.
    
    Parameters:
    df (DataFrame): The dataframe containing the data.
    column (str): The column name to plot.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column])
    plt.title(title)
    plt.show()

plot_boxplot(train_data, 'Age', 'Boxplot of Age in Train Data')
plot_boxplot(test_data, 'Age', 'Boxplot of Age in Test Data')
plot_boxplot(train_data, 'Fare', 'Boxplot of Fare in Train Data')
plot_boxplot(test_data, 'Fare', 'Boxplot of Fare in Test Data')

# Handling missing values
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# Remove rows where 'Cabin' is missing
train_data.dropna(subset=['Cabin'], inplace=True)
test_data.dropna(subset=['Cabin'], inplace=True)

# Fill missing values in 'Embarked' using the mode (most common value)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print("Train data:\n", train_data.isnull().sum())
print("Test data:\n", test_data.isnull().sum())

# Identifying and removing outliers
def remove_outliers(df, column, threshold=3):
    """
    Removes outliers from the specified column in the dataframe based on z-scores.
    
    Parameters:
    df (DataFrame): The dataframe containing the data.
    column (str): The column name to remove outliers from.
    threshold (float): The z-score threshold to identify outliers.
    
    Returns:
    DataFrame: The dataframe with outliers removed.
    """
    df = df.dropna(subset=[column])  # Ensure no NaN values
    z_scores = stats.zscore(df[column])
    return df[(z_scores < threshold) & (z_scores > -threshold)]

# Remove outliers for Age and Fare in both train and test data
cleaned_train = remove_outliers(train_data, 'Age')
cleaned_train = remove_outliers(cleaned_train, 'Fare')

cleaned_test = remove_outliers(test_data, 'Age')
cleaned_test = remove_outliers(cleaned_test, 'Fare')

# Plot boxplots after cleaning
plot_boxplot(cleaned_train, 'Age', 'Boxplot of Age in Train Data (After Cleaning)')
plot_boxplot(cleaned_train, 'Fare', 'Boxplot of Fare in Train Data (After Cleaning)')

# Saving cleaned datasets
cleaned_train.to_csv(cleaned_train_file_path, index=False)
cleaned_test.to_csv(cleaned_test_file_path, index=False)

print("\n✅ Cleaned datasets saved as 'cleaned_train_data.csv' and 'cleaned_test_data.csv'.")
