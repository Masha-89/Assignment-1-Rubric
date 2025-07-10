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
        
# Check for missing values
# This function will be used to clean the data by filling missing values and encoding categorical variables.

class DataCleaner:
    def __init__(self, df):
        self.df = df
    
    def Values_Missing(self):
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            print("Missing values found in the dataset.")
        else:
            print("No missing values found, dataset is filled.")
        return missing_values
    
    def duplicates_verification(self):
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows in the dataset.")
            self.df.drop_duplicates(inplace=True)
            print("Duplicates removed.")
        else:
            print("No duplicate rows found.")

    def Validate_data_Range(self):
     
        if 'Age' in self.df.columns:
            invalid_age = self.df[(self.df['Age'] < 0) | (self.df['Age'] > 100)]
            if not invalid_age.empty:
                print(f"Invalid ages found:\n{invalid_age}")
                self.df = self.df[(self.df['Age'] >= 0) & (self.df['Age'] <= 100)]
                print("Invalid ages removed.")
            else:
                print("All ages are valid.")
        else:
            print("'Age' column not found in the dataset.")
    
# This class will be used to analyze student performance based on their habits and other features.
class StudentPerformanceAnalysis:
    def __init__(self, df):
        self.df = df

        def mean_median_study_by_mental_health(self):
            if 'mental_health_ratin' not in self.df.columns or 'study_hours_per_day' not in self.df.columns:
                group = self.df.groupby('mental_health_ratin')['study_hours_per_day'].agg(['mean', 'median'])
                print("Mean and median study hours per day by mental health rating:")
                print(group)

            else:
                print("Required columns are missing for analysis.")
                return None
            
        def exam_sleep_correlation(self):
            if 'exam_score' in self.df.columns and 'sleep_hours' in self.df.columns:
             correlation = self.df['exam_score'].corr(self.df['sleep_hours'])
             print(f"Correlation between exam scores and sleeping hours: {correlation}")
            else:
                print("Required columns are missing for correlation analysis.")
            return None
        
        def scoial_media_outliers(self):
            if 'social_media_hours' in self.df.columns:
                Q1 = self.df['social_media_hours'].quantile(0.25)
                Q3 = self.df['social_media_hours'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df['social_media_hours'] < lower_bound) |
                                   (self.df['social_media_hours'] > upper_bound)]
                if not outliers.empty:
                    print(f"Outliers in social media hours:\n{outliers}")   
            else:
                print("Social media hours column not found.")
                return None