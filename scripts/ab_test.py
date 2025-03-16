import pandas as pd
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

def ab_test_numeric(group_A, group_B, metric_name):
      """
      Perform an independent t-test on two numeric groups.
    
      Parameters:
        group_A (pd.Series): Control group data.
        group_B (pd.Series): Test group data.
        metric_name (str): Name of the metric being tested.
    
      Returns:
        None: Prints the t-test results.
      """
      t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=False)

      print(f"\nA/B Test for {metric_name}:")
      print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")

      if p_value < 0.05:
        print("✅ Reject Null Hypothesis: The groups are significantly different.")
      else:
        print("❌ Fail to Reject Null Hypothesis: No significant difference between groups.")

def perform_ab_tests(df):
    """
    Perform A/B tests by automatically detecting and comparing groups.
    
    Parameters:
        df (pd.DataFrame): The dataset.
    """
    print("\nStarting A/B Hypothesis Testing...")

    # Check for missing values before proceeding
    check_missing_values(df)
     ## Hypothesis 1: Risk differences across provinces
    if 'Province' in df.columns and 'TotalClaims' in df.columns:
        provinces = df['Province'].unique()
        if len(provinces) > 1:
            print("\nTesting Risk Differences Across Provinces:")
            for i in range(len(provinces)):
                for j in range(i + 1, len(provinces)):
                    group_A = df[df['Province'] == provinces[i]]['TotalClaims']
                    group_B = df[df['Province'] == provinces[j]]['TotalClaims']
                    print(f"\nComparing {provinces[i]} vs {provinces[j]}:")
                    ab_test_numeric(group_A, group_B, "Total Claims Across Provinces")
def ab_test_zip(df):
 ## Hypothesis 3: Profit margin differences between zip codes
   df['ProfitMargin'] = df['TotalPremium'] - df['TotalClaims']
   print("✅ Profit Margin calculated and added to the dataset.")
   if 'Province' in df.columns and 'ProfitMargin' in df.columns:
        zip_codes = df['Province'].unique()
        if len(zip_codes) > 1:
            print("\nTesting Profit Margin Differences Between Zip Codes:")
            for i in range(len(zip_codes)):
                for j in range(i + 1, len(zip_codes)):
                    group_A = df[df['Province'] == zip_codes[i]]['ProfitMargin']
                    group_B = df[df['Province'] == zip_codes[j]]['ProfitMargin']
                    print(f"\nComparing {zip_codes[i]} vs {zip_codes[j]}:")
                    ab_test_numeric(group_A, group_B, "Profit Margin Between province Codes")
   else:
       print("Zip_code doesn't Exit in the datase")
def ab_risk_zip(df):
  ## Hypothesis 2: Risk differences between zip codes
    if 'ZipCode' in df.columns and 'TotalClaims' in df.columns:
        zip_codes = df['ZipCode'].unique()
        if len(zip_codes) > 1:
            print("\nTesting Risk Differences Between Zip Codes:")
            for i in range(len(zip_codes)):
                for j in range(i + 1, len(zip_codes)):
                    group_A = df[df['ZipCode'] == zip_codes[i]]['TotalClaims']
                    group_B = df[df['ZipCode'] == zip_codes[j]]['TotalClaims']
                    print(f"\nComparing {zip_codes[i]} vs {zip_codes[j]}:")
                    ab_test_numeric(group_A, group_B, "Total Claims Between Zip Codes")
def ab_risk_gen(df):
 ## Hypothesis 4: Risk differences between Women and Men
    if 'Gender' in df.columns and 'TotalClaims' in df.columns:
        genders = ['Male','Female']
        if len(genders) > 1:  # Assuming only 'Male' and 'Female'
            print("\nTesting Risk Differences Between Genders:")
            group_A = df[df['Gender'] == genders[0]]['TotalClaims']
            group_B = df[df['Gender'] == genders[1]]['TotalClaims']
            print(f"\nComparing {genders[0]} vs {genders[1]}:")
            ab_test_numeric(group_A, group_B, "Total Claims Between Genders")

    print("\n✅ A/B Testing Completed.")
  