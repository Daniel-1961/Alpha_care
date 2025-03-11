import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind

def load_data(file_path):
    """
    Load the dataset into a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        df['province'] = df['province'].astype(str)
        df['zip_code'] = df['zip_code'].astype(str)
        df['gender'] = df['gender'].astype(str)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def chi_square_test(df, column):
    """
    Performs a Chi-Square test to check risk differences across categories.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        column (str): The categorical column to test (province, zip_code, gender).
    
    Prints:
        The p-value and whether to reject or fail to reject the null hypothesis.
    """
    if column not in df.columns:
        print(f"Column {column} not found in dataset.")
        return

    contingency_table = pd.crosstab(df[column], df['fraud_flag'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-Square Test for {column}:")
    print(f"p-value: {p}")
    
    if p < 0.05:
        print(f"❌ Reject Null Hypothesis: Significant risk differences across {column}.")
    else:
        print(f"✅ Fail to Reject Null Hypothesis: No significant risk differences across {column}.")

def t_test(df, group_col, metric_col):
    """
    Performs an independent T-Test to check margin (profit) differences between groups.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        group_col (str): The categorical column to group by (e.g., zip_code).
        metric_col (str): The numerical column to compare (e.g., profit_margin).
    
    Prints:
        The p-value and whether to reject or fail to reject the null hypothesis.
    """
    if group_col not in df.columns or metric_col not in df.columns:
        print(f"Columns {group_col} or {metric_col} not found in dataset.")
        return

    unique_groups = df[group_col].unique()
    
    if len(unique_groups) < 2:
        print(f"Not enough unique values in {group_col} for t-test.")
        return
    
    group1 = df[df[group_col] == unique_groups[0]][metric_col]
    group2 = df[df[group_col] == unique_groups[1]][metric_col]
    
    t_stat, p = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
    
    print(f"\nT-Test for {metric_col} across {group_col}:")
    print(f"p-value: {p}")
    
    if p < 0.05:
        print(f"❌ Reject Null Hypothesis: Significant profit margin differences across {group_col}.")
    else:
        print(f"✅ Fail to Reject Null Hypothesis: No significant profit margin differences across {group_col}.")
