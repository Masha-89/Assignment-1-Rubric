import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('student_habits_performance.csv')
# Display the first few rows of the DataFrame
print(df.head())