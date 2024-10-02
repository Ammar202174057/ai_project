Project Overview
This project involves predicting calorie expenditure based on exercise data using a machine learning model. The dataset consists of exercise details and corresponding calorie counts. The goal is to build a regression model to predict calories based on given features.

Code Explanation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```
- **Import Libraries**: Import necessary libraries for data manipulation, visualization, and machine learning.

```python
calories = pd.read_csv('calories.csv')
exercise_data = pd.read_csv('exercise.csv')
```
- **Load Data**: Read CSV files into Pandas DataFrames.

```python
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
```
- **Combine Data**: Concatenate exercise data with calorie information.

```python
calories_data.shape
calories_data.info()
```
- **Data Overview**: Check the shape and summary information of the dataset.

```python
calories_data.isnull().sum()
```
- **Check Missing Values**: Identify any missing values in the dataset.

```python
calories_data.describe()
```
- **Statistical Summary**: Get statistical measures like mean and standard deviation.

```python
sns.set()
sns.histplot(calories_data['Gender'])
sns.distplot(calories_data['Age'])
sns.distplot(calories_data['Height'])
sns.distplot(calories_data['Weight'])
```
- **Data Visualization**: Plot distributions for gender, age, height, and weight.

```python
correlation = calories_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
```
- **Correlation Heatmap**: Visualize correlations between features using a heatmap.

```python
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
```
- **Encode Categorical Data**: Convert gender from categorical to numerical values.

```python
X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']
```
- **Define Features and Target**: Separate features (`X`) and target variable (`Y`).

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```
- **Split Data**: Divide data into training and test sets.

```python
model = XGBRegressor()
model.fit(X_train, Y_train)
```
- **Train Model**: Initialize and train an XGBoost Regressor model on the training data.

```python
test_data_prediction = model.predict(X_test)
```
- **Make Predictions**: Predict calories for the test data.

```python
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)
```
- **Evaluate Model**: Calculate and print the Mean Absolute Error of the model.

```python
input_data = (0,68,190.0,94.0,29.0,105.0,40.8)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
```
- **New Prediction**: Create a sample input, reshape it, and predict calories using the trained model.
