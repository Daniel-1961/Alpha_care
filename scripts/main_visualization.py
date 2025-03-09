import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceEda:
    def __init__(self, df):
        """Initialize the class with the dataset."""
        self.df = df

    def ensure_column_exists(self, column_name, default_values=None):
        """Ensure a specific column exists in the dataset; if not, create a default."""
        if column_name not in self.df.columns:
            if default_values:
                self.df[column_name] = np.random.choice(default_values, size=len(self.df))
                print(f"{column_name} column added with dummy values: {default_values}")
            else:
                print(f"Warning: {column_name} is missing and no default values were provided.")

    def ensure_datetime_column(self, date_column):
        """Ensure the given column is in datetime format."""
        if date_column in self.df.columns:
            self.df[date_column] = pd.to_datetime(self.df[date_column], errors='coerce')
            print(f"{date_column} converted to datetime format.")
        else:
            print(f"Warning: {date_column} not found in dataset.")

    def plot_premium_claim_trends(self):
        """Plot TotalPremium and TotalClaims trends over time for a sample ZipCode."""
        self.ensure_column_exists('ZipCode', ['Z1', 'Z2', 'Z3'])
        self.ensure_datetime_column('TransactionMonth')

        required_columns = {'TotalPremium', 'TotalClaims', 'ZipCode', 'TransactionMonth'}
        if not required_columns.issubset(self.df.columns):
            print(f"Error: Required columns missing: {required_columns - set(self.df.columns)}")
            return

        # Group by ZipCode and TransactionMonth, then calculate monthly means
        monthly_grouped = self.df.groupby(['ZipCode', 'TransactionMonth'])[['TotalPremium', 'TotalClaims']].mean().reset_index()

        # Choose the most frequent ZipCode
        sample_zip = self.df['ZipCode'].value_counts().idxmax()
        zip_data = monthly_grouped[monthly_grouped['ZipCode'] == sample_zip]

        # Plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=zip_data, x='TransactionMonth', y='TotalPremium', label='TotalPremium', color='blue', marker='o', linewidth=2)
        sns.lineplot(data=zip_data, x='TransactionMonth', y='TotalClaims', label='TotalClaims', color='orange', marker='s', linewidth=2)
        
        plt.title(f'TotalPremium and TotalClaims Trends for ZipCode: {sample_zip}', fontsize=14, fontweight='bold')
        plt.xlabel('Transaction Month', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def plot_claims_by_vehicle_province(self):
        """Plot Total Claims by Vehicle Type Across Provinces."""
        self.ensure_column_exists('Province', ['Province_A', 'Province_B', 'Province_C'])
        self.ensure_column_exists('VehicleType', ['Sedan', 'SUV', 'Truck'])

        required_columns = {'Province', 'VehicleType', 'TotalClaims'}
        if not required_columns.issubset(self.df.columns):
            print(f"Error: Required columns missing: {required_columns - set(self.df.columns)}")
            return

        # Group data by Province and VehicleType
        covergroup_geography = self.df.groupby(['Province', 'VehicleType'])['TotalClaims'].sum().reset_index()

        # Plot
        plt.figure(figsize=(14, 8))
        sns.barplot(data=covergroup_geography, x='Province', y='TotalClaims', hue='VehicleType', palette='viridis')

        plt.title('Total Claims by Vehicle Types Across Provinces', fontsize=14, fontweight='bold')
        plt.xlabel('Province', fontsize=12)
        plt.ylabel('Total Claims', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
def plot_average_claim_trends(self):
    """Plot the average claim trends over time across provinces."""
    self.ensure_column_exists('Province', ['Province_A', 'Province_B', 'Province_C'])
    self.ensure_datetime_column('TransactionMonth')

    required_columns = {'Province', 'TransactionMonth', 'TotalClaims'}
    if not required_columns.issubset(self.df.columns):
        print(f"Error: Required columns missing: {required_columns - set(self.df.columns)}")
        return

    # Group by Province and TransactionMonth, then calculate the mean TotalClaims
    premium_geography = self.df.groupby(['Province', 'TransactionMonth'])['TotalClaims'].mean().reset_index()

    # Plot
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=premium_geography,
        x='TransactionMonth',
        y='TotalClaims',
        hue='Province',
        palette='Dark2',
        marker="o",
        linewidth=2
    )

    # Customizing the plot for better aesthetics
    plt.title('Average Claim Trends Across Provinces Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Transaction Month', fontsize=12)
    plt.ylabel('Average Total Claims', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Province', fontsize=12, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# Example Usage:
# eda = InsuranceEDA('data.csv')
# eda.plot_premium_claim_trends()
# eda.plot_claims_by_vehicle_province()
