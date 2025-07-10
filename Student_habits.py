import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv('student_habits_performance.csv')
# Display the first few rows of the DataFrame
# print(df.head())

# This class will be used to encapsulate the functionality of loading data from a CSV file.
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
            return df
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None