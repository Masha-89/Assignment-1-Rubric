import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os
import pickle 
import markdown
import pdfkit 

# This script is designed to analyze student habits and their impact on academic performance.
df = pd.read_csv('student_habits_performance.csv', delimiter = ';')
# Display the first few rows of the DataFrame
# print(df.head())
# print(df.columns)

# This class will be used to encapsulate the functionality of loading data from a CSV file.
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
    def load_data(self):
        try:
            df = pd.read_csv(self.file_path, delimiter=';')
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
     
        if 'age' in self.df.columns:
            invalid_age = self.df[(self.df['age'] < 0) | (self.df['age'] > 100)]
            if not invalid_age.empty:
                print(f"Invalid ages found:\n{invalid_age}")
                self.df = self.df[(self.df['age'] >= 0) & (self.df['age'] <= 100)]
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
            if 'mental_health_rating' in self.df.columns or 'study_hours_per_day' in self.df.columns:
               grouped = self.df.groupby('mental_health_rating')['study_hours_per_day']
               group = grouped.agg(['mean', 'median']).reset_index()
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

# This class will be used to visualize the data and the results of the analysis.
class VisualizationEngine:
    def __init__(self, df):
        self.df = df
# Plots a scatter plot of sleep hours vs. exam scores.
    def study_time_histogram(self):
        if 'study_hours_per_day' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df['study_hours_per_day'], bins=25, kde=True)
            plt.title('Distribution of Study Hours per Day')
            plt.xlabel('Study Hours per Day')
            plt.ylabel('Frequency')
            plt.show()
        else:
            print("Study hours column not found in Dataset.")

# Plots a scatter plot of sleep hours vs. exam scores.
    def sleep_vs_exam_scores(self):
        if 'sleep_hours' in self.df.columns and 'exam_score' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='sleep_hours', y='exam_score', data=self.df)
            plt.title('Sleep Hours vs Exam Scores')
            plt.xlabel('Sleep Hours')
            plt.ylabel('Exam Score')
            plt.show()
        else:
            print("Required columns are missing for sleep vs exam scores analysis.")
            return None
    
# Plots boxplots of exam scores grouped by diet quality.
    def scores_by_diet_quality(self):
        if 'diet_quality' in self.df.columns and 'exam_score' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='diet_quality', y='exam_score', data=self.df)
            plt.title('Exam Scores by Diet Quality')
            plt.xlabel('Diet Quality')
            plt.ylabel('Exam Score')
            plt.show()

        else:
            print("Required columns are missing for diet quality analysis.")
            return None

class ScorePredictor:
    def __init__(self, df):
        self.df = df
        self.model = LinearRegression()
        self.label_encoder = LabelEncoder()
        self.fitted = False

    def train_model(self, feature_col, target_col):
        try:
            for col in feature_col + [target_col]:
                if col in self.df.columns:
                    if self.df[col].dtype == 'object':
                        self.df[col] = self.label_encoder.fit_transform(self.df[col])
            self.model = LinearRegression()
            X = self.df[feature_col]
            y = self.df[target_col]
            self.model.fit(X, y)
            self.fitted = True
            print("Model trained successfully.")
        except Exception as e:
            print(f"An error occurred while training the model: {e}")

    def save_model(self, file_path):
        if not self.fitted:
            print("Error: Cannot save an untrained model.")
            return
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                self.model = pickle.load(f)
            self.fitted = True
            print(f"Model loaded from {file_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

class ReportExporter:
    def __init__(self, output_dir='report'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.content = "# Student Analysis Report\n\n"

    def add_section(self, title, text):
        self.content += f"## {title}\n\n{text}\n\n"

    def add_image(self, image_file, caption=""):
        self.content += f"![{caption}]({image_file})\n\n"

    def save_markdown(self, filename='report.md'):
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.content)
            print(f"Markdown saved: {path}")
        except PermissionError:
            print(f"Permission denied writing Markdown: {path}")

    def export_pdf(self, md_file='report.md', pdf_file='report.pdf'):
        md_path = os.path.join(self.output_dir, md_file)
        pdf_path = os.path.join(self.output_dir, pdf_file)
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                html = markdown.markdown(f.read())
            pdfkit.from_string(html, pdf_path)
            print(f"PDF saved: {pdf_path}")
        except PermissionError:
            print(f"Permission denied writing PDF: {pdf_path}")
    
# Example usage
if __name__ == "__main__":
    loader = DataLoader('student_habits_performance.csv')
    df = loader.load_data()
    if df is not None:
        cleaner = DataCleaner(df)
        cleaner.Values_Missing()
        cleaner.duplicates_verification()
        cleaner.Validate_data_Range()

        analysis = StudentPerformanceAnalysis(df)
        analysis.mean_median_study_by_mental_health()
        analysis.exam_sleep_correlation()
        analysis.scoial_media_outliers()

        visualizer = VisualizationEngine(df)
        visualizer.study_time_histogram()
        visualizer.sleep_vs_exam_scores()
        visualizer.scores_by_diet_quality()

        predictor = ScorePredictor(df)
        predictor.train_model(['study_hours_per_day', 'sleep_hours'], 'exam_score')

        report_exporter = ReportExporter()
        report_exporter.add_section("Data Summary", "This section contains a summary of the data.")
        print("Markdown saved. Looking for PDF export next...")
        report_exporter.save_markdown()
        print("Attempting PDF export...")
        report_exporter.export_pdf()
        print("Done. Check the /report folder for output.")
        print("Analysis and report generation completed.")
    else:   
        print("Data loading failed. Please check the file path and format.")


        import webbrowser
        webbrowser.open(os.path.join('report', 'report.pdf'))
# End of the code. 