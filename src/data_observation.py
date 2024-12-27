import pandas as pd
df=pd.read_csv("..\\Data\\output.csv", low_memory=False)
numerical_col=['TotalPremium','TotalClaims']
numerical_summery=df[numerical_col].describe()
print(numerical_summery)

variability = df[numerical_col].agg(['std', 'var', 'min', 'max', lambda x: x.max() - x.min()])
variability.rename(index={'<lambda>': 'range'}, inplace=True)

print("\nVariability:\n", variability)
