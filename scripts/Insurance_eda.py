import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceEDA:
    def __init__(self, df):
        """Initialize the class with the dataset."""
        self.df =df
    
    def display_info(self):
        """Display basic information about the dataset."""
        print("\nDataset Info:")
        self.df.info()
        print("\nFirst 5 Rows:")
        print(self.df.head())
    
    def check_missing_values(self):
        """Check for missing values in the dataset."""
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
    
    def handle_missing_values(self):
      """Handle missing values in the dataset."""
      missing_values = self.df.isnull().sum()
      missing_cols = missing_values[missing_values > 0].index.tolist()
      #print("missing col are"+missing_cols)
      if not missing_cols:
        print("\nNo missing values detected.")
        return

      print("\nHandling Missing Values:")

      for col in missing_cols:
        if col not in self.df.columns:
            print(f"Warning: Column '{col}' not found in dataset. Skipping...")
            continue  # Skip missing columns to prevent KeyError

        if self.df[col].dtype == 'object':  # Categorical column
            self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)  # Fill with mode
        elif np.issubdtype(self.df[col].dtype, np.number):  # Numerical column
            self.df[col].fillna(self.df[col].median(), inplace=True)  # Fill with median
        else:
            print(f"Skipping column {col}, unsupported data type: {self.df[col].dtype}")

    print("\nMissing values handled successfully.")

    
    def convert_dates(self, date_columns):
        """Convert specified columns to datetime format."""
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        print("\nDate columns converted.")
    
    def descriptive_statistics(self, num_columns):
        """Calculate and display descriptive statistics for numerical columns."""
        print("\nDescriptive Statistics:")
        print(self.df[num_columns].describe())
    
    def detect_outliers(self, col):
        """Detect outliers using the IQR method for a single column."""
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))]
        return outliers.shape[0]
    
    def check_outliers(self, num_columns):
        """Check for outliers in multiple numerical columns."""
        print("\nOutlier Counts:")
        outlier_counts = {col: self.detect_outliers(col) for col in num_columns}
        print(outlier_counts)
        
        for col in num_columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(y=self.df[col])
            plt.title(f'Box Plot of {col}')
            plt.show()
    
    def detect_categorical_columns(self, threshold=20):
        """Detect categorical columns based on unique value count."""
        cat_columns = [col for col in self.df.columns if self.df[col].dtype == 'object' or self.df[col].nunique() <= threshold]
        print("\nDetected Categorical Columns:", cat_columns)
        return cat_columns
    
    def convert_categorical(self, categorical_columns):
        """Convert specified columns to categorical type."""
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        print("\nCategorical columns converted.")
    
    def plot_distributions(self, num_columns, cat_columns):
        """Plot histograms for numerical columns and bar charts for categorical columns."""
        for col in num_columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col], bins=30, kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()
    

        
        for col in cat_columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.df, x=col, order=self.df[col].value_counts().index)
            plt.title(f'Bar Chart of {col}')
            plt.xticks(rotation=45)
            plt.show()
    def correlation_matrix(self):
     """Plots the correlation matrix as a heatmap"""
     numerical_df =self.df.select_dtypes(include=['number'])  # Select only numerical columns
     correlation_matrix = numerical_df.corr()
     plt.figure(figsize=(12, 8))
     sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
     plt.title("Correlation Matrix")
     plt.show()
    def run_eda(self):
        """Run all EDA steps."""
        self.display_info()
        self.check_missing_values()
        self.handle_missing_values()
        self.convert_dates(['TransactionMonth', 'VehicleIntroDate'])
        self.descriptive_statistics(['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm'])
        self.check_outliers(['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm'])
        detected_cat_columns = self.detect_categorical_columns()
        self.convert_categorical(detected_cat_columns)
        self.self.detect_categorical_columns()
        self.correlation_analysis()
        print("\nEDA completed.")

# Example Usage
# eda = InsuranceEDA('data.csv')
# eda.run_eda()
