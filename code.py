import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the data
calories = pd.read_csv('calories.csv')
exercise_data = pd.read_csv('exercise.csv')
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = XGBRegressor()
model.fit(X_train, Y_train)

# GUI setup
def predict_calories():
    try:
        gender = int(entry_gender.get())
        age = float(entry_age.get())
        height = float(entry_height.get())
        weight = float(entry_weight.get())
        duration = float(entry_duration.get())
        heart_rate = float(entry_heart_rate.get())
        body_temp = float(entry_body_temp.get())

        input_data = (gender, age, height, weight, duration, heart_rate, body_temp)
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        prediction = model.predict(input_data_as_numpy_array)
        
        messagebox.showinfo("Prediction", f"Estimated Calories: {prediction[0]:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers")

app = tk.Tk()
app.title("Calorie Prediction")

tk.Label(app, text="Gender (0=Male, 1=Female):").grid(row=0, column=0)
entry_gender = tk.Entry(app)
entry_gender.grid(row=0, column=1)

tk.Label(app, text="Age:").grid(row=1, column=0)
entry_age = tk.Entry(app)
entry_age.grid(row=1, column=1)

tk.Label(app, text="Height (cm):").grid(row=2, column=0)
entry_height = tk.Entry(app)
entry_height.grid(row=2, column=1)

tk.Label(app, text="Weight (kg):").grid(row=3, column=0)
entry_weight = tk.Entry(app)
entry_weight.grid(row=3, column=1)

tk.Label(app, text="Duration (min):").grid(row=4, column=0)
entry_duration = tk.Entry(app)
entry_duration.grid(row=4, column=1)

tk.Label(app, text="Heart Rate:").grid(row=5, column=0)
entry_heart_rate = tk.Entry(app)
entry_heart_rate.grid(row=5, column=1)

tk.Label(app, text="Body Temp (°C):").grid(row=6, column=0)
entry_body_temp = tk.Entry(app)
entry_body_temp.grid(row=6, column=1)

tk.Button(app, text="Predict", command=predict_calories).grid(row=7, column=0, columnspan=2)

app.mainloop()