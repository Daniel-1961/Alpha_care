import pandas as pd
import numpy as np

class InsuranceEDA:
    def __init__(self, df):
        """Initialize the class with the dataset."""
        self.df = df
    
    def display_info(self):
        """Display basic information about the dataset."""
        print("\nDataset Info:")
        self.df.info()
        print("\nFirst 5 Rows:")
        print(self.df.head())
    
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
    
    def convert_categorical(self, categorical_columns):
        """Convert specified columns to categorical type."""
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        print("\nCategorical columns converted.")
    
    def run_eda(self):
        """Run all EDA steps."""
        self.display_info()
        self.convert_dates(['TransactionMonth', 'VehicleIntroDate'])
        self.descriptive_statistics(['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm'])
        self.check_outliers(['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm'])
        self.convert_categorical(['Citizenship', 'LegalType', 'Gender', 'Province', 'CoverCategory', 'VehicleType'])
        print("\nEDA completed.")

# Example Usage
# eda = InsuranceEDA('data.csv')
# eda.run_eda()
