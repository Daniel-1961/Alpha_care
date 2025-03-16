import pandas as pd
import numpy as np
from scipy import stats

def load_data(file_path):
    """
    Load dataset into a Pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def check_missing_values(df):
    """
    Print the number of missing values per column.
    
    Parameters:
        df (pd.DataFrame): The dataset.
    """
    print("\nMissing Values:")
    print(df.isnull().sum())

def ab_test_numeric(df, metric_name):
    """
    Perform an independent t-test on two numeric groups.
    
    Parameters:
        group_A (pd.Series): Control group data.
        group_B (pd.Series): Test group data.
        metric_name (str): Name of the metric being tested.
    
    Returns:
        None: Prints the t-test results.
    """
    if 'Province' in df.columns and 'TotalClaims' in df.columns:
        provinces = df['Province'].unique()
        if len(provinces) > 1:
            for i in range(len(provinces)):
                for j in range(i + 1, len(provinces)):
                    group_A = df[df['Province'] == provinces[i]]['TotalClaims']
                    group_B = df[df['Province'] == provinces[j]]['TotalClaims']
                    print(f"\nComparing {provinces[i]} vs {provinces[j]}:")
                    print(f"Group A: {group_A.tolist()}")
                    print(f"Group B: {group_B.tolist()}")
                    t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=False)
                    print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")

                    if p_value < 0.05:
                     print("✅ Reject Null Hypothesis: The groups are significantly different.")
                    else:
                       print("❌ Fail to Reject Null Hypothesis: No significant difference between groups.")

def ab_test_categorical(df, column_name, target_column):
    """
    Perform a chi-square test for independence between two categorical variables.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        column_name (str): Categorical variable to test.
        target_column (str): Outcome variable (e.g., Fraud or Claims).
    
    Returns:
        None: Prints the chi-square test results.
    """
    contingency_table = pd.crosstab(df[column_name], df[target_column])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\nChi-Square Test for {column_name} vs {target_column}:")
    print(f"Chi-Square: {chi2:.4f}, P-Value: {p_value:.4f}")

    if p_value < 0.05:
        print("✅ Reject Null Hypothesis: The variables are dependent.")
    else:
        print("❌ Fail to Reject Null Hypothesis: No significant relationship.")

def perform_ab_tests(df):
    """
    Perform A/B hypothesis tests on insurance data.
    
    Parameters:
        df (pd.DataFrame): The dataset.
    
    Returns:
        None: Prints results of multiple A/B tests.
    """
    print("\nStarting A/B Hypothesis Testing...")

    # Check for missing values before proceeding
    check_missing_values(df)

    ## Hypothesis 1: Risk differences across provinces
    if 'Province' in df.columns and 'TotalClaims' in df.columns:
        provinces = df['Province'].unique()
        if len(provinces) > 1:
            group_A = df[df['Province'] == provinces[0]]['TotalClaims']
            group_B = df[df['Province'] == provinces[1]]['TotalClaims']
    ab_test_numeric(group_A, group_B, "Total Claims Across Provinces")

    ## Hypothesis 2: Risk differences between zip codes
    if 'ZipCode' in df.columns and 'TotalClaims' in df.columns:
        zip_codes = df['ZipCode'].unique()
        if len(zip_codes) > 1:
            group_A = df[df['ZipCode'] == zip_codes[0]]['TotalClaims']
            group_B = df[df['ZipCode'] == zip_codes[1]]['TotalClaims']
    #ab_test_numeric(group_A, group_B, "Total Claims Between Zip Codes")

    ## Hypothesis 3: Profit margin differences between zip codes
    if 'ZipCode' in df.columns and 'ProfitMargin' in df.columns:
        zip_codes = df['ZipCode'].unique()
        if len(zip_codes) > 1:
            group_A = df[df['ZipCode'] == zip_codes[0]]['ProfitMargin']
            group_B = df[df['ZipCode'] == zip_codes[1]]['ProfitMargin']
   # ab_test_numeric(group_A, group_B, "Profit Margin Between Zip Codes")

    ## Hypothesis 4: Risk differences between Women and Men
    if 'Gender' in df.columns and 'TotalClaims' in df.columns:
        if df['Gender'].nunique() == 2:
            group_A = df[df['Gender'] == 'Male']['TotalClaims']
            group_B = df[df['Gender'] == 'Female']['TotalClaims']
#ab_test_numeric(group_A, group_B, "Total Claims Between Men and Women")
    
    print("\n✅ A/B Testing Completed.")

