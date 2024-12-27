import pandas as pd

input_file = '../Data/raw/MachineLearningRating_v3.txt'
output_file = '../Data/raw/MachineLearningRating_v3.csv'

df = pd.read_csv(input_file, delimiter='|')  # Use '|' as the delimiter

# Save the parsed data to a new .csv file
df.to_csv(output_file, index=False)

print(f"File successfully parsed and saved as {output_file}")
print(df.head())
print(df.info())

print(f"File converted and saved as {output_file}")
