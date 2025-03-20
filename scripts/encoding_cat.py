import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
class FeatureEngineering:
    def __init__(self, df, target_variable, high_cardinality_threshold=50):
        """
        Initialize the FeatureEngineering class.
        :param df: DataFrame containing the dataset.
        :param target_variable: Name of the target variable column.
        :param high_cardinality_threshold: Threshold for high-cardinality features.
        """
        self.df = df
        self.target_variable = target_variable
        self.high_cardinality_threshold = high_cardinality_threshold

    def handle_missing_values(self, cols):
        """Fills missing values with a placeholder 'Unknown'."""
        for col in cols:
            self.df[col] = self.df[col].fillna("Unknown")

    def drop_unnecessary_columns(self, cols_to_drop):
        """Drops columns that are not relevant for training."""
        print(f"Dropping columns: {cols_to_drop}")
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')  # `errors='ignore'` handles non-existent columns gracefully
    
    def create_vehicle_age(self):
        """Creates a new feature 'VehicleAge' based on 'RegistrationYear'."""
        current_year = datetime.now().year
        self.df['VehicleAge'] = current_year - self.df['RegistrationYear']
        self.df.drop(columns=['RegistrationYear'], inplace=True)

    def encode_high_cardinality(self):
        """Encodes high-cardinality features using target encoding."""
        for col in self.df.select_dtypes(include=['object']).columns:
            unique_vals = self.df[col].nunique()
            if unique_vals > self.high_cardinality_threshold:
                print(f"Encoding {col} using target encoding (high cardinality: {unique_vals} unique values)")
                target_encoder = ce.TargetEncoder(cols=[col])
                self.df[col] = target_encoder.fit_transform(self.df[col], self.df[self.target_variable])

    def frequency_encode(self, cols):
        """Applies frequency encoding to selected columns."""
        for col in cols:
            self.df[col] = self.df[col].map(self.df[col].value_counts(normalize=True))

    def label_encode(self, cols):
        """Applies Label Encoding or Ordinal Encoding for low-cardinality features."""
        for col in cols:
            print(f"Encoding {col} using Label Encoding")
            label_encoder = LabelEncoder()
            self.df[col] = label_encoder.fit_transform(self.df[col].astype(str))

    def log_transform(self, cols):
        """Applies log transformation to selected columns to handle skewness."""
        for col in cols:
            self.df[col] = np.log1p(self.df[col])

    def create_interaction_features(self):
        """Creates new interaction features."""
        # Premium per insured amount
        self.df['PremiumPerInsured'] = self.df['CalculatedPremiumPerTerm'] / self.df['SumInsured']
        self.df['PremiumPerInsured'].replace([float('inf'), -float('inf')], 0, inplace=True)
        self.df['PremiumPerInsured'].fillna(0, inplace=True)
        """Creates new features from existing numerical and date-based columns."""
        if 'TotalPremium' in self.df.columns and 'TotalClaims' in self.df.columns:
            self.df['ClaimRatio'] = self.df['TotalClaims'] / (self.df['TotalPremium'] + 1e-5)
            self.df['PremiumPerClaim'] = self.df['TotalPremium'] / (self.df['TotalClaims'] + 1e-5)

        if 'PolicyStartDate' in self.df.columns:
            self.df['PolicyStartDate'] = pd.to_datetime(self.df['PolicyStartDate'])
            self.df['PolicyStartYear'] = self.df['PolicyStartDate'].dt.year
            self.df['PolicyStartMonth'] = self.df['PolicyStartDate'].dt.month
            self.df['PolicyElapsedDays'] = (pd.to_datetime('today') - self.df['PolicyStartDate']).dt.days
    import pandas as pd


    def one_hot_encode(self, columns_to_encode):
        """
        Performs One-Hot Encoding on specified categorical columns.

        Parameters:
        columns_to_encode (list): List of categorical columns to be one-hot encoded.

        Returns:
        None: Modifies self.df in place.
        """
        # Apply One-Hot Encoding
        encoded_df = pd.get_dummies(self.df, columns=columns_to_encode, drop_first=True)

        # Update the dataframe
        self.df = encoded_df


    def binary_encode(self, cols):
     """Encodes binary features as integers, handling non-binary data gracefully."""
     for col in cols:
        # Check for unique values in the column
        unique_values = self.df[col].unique()
        
        # If the column contains non-binary values, map them appropriately
        if set(unique_values) != {0, 1}:  # If not already binary
            print(f"Handling non-binary values in column: {col}")
            
            # Example mapping logic (customize as needed for your data)
            mapping = {
                "Yes": 1, "No": 0,
                "True": 1, "False": 0,
                "More than 6 months": 1, "Less than 6 months": 0,
                np.nan: 0  # Handle NaN as a default value
            }
            self.df[col] = self.df[col].map(mapping).fillna(0).astype(int)  # Replace unmapped values with 0
        else:
            # Directly convert binary columns to integers
            self.df[col] = self.df[col].astype(int)

     print("Feature engineering complete.")
     return self.df


    def process(self, cols_to_drop=None):
        """Executes all feature engineering steps."""
        # Drop unnecessary columns
        if cols_to_drop:
            self.drop_unnecessary_columns(cols_to_drop)
        
        # Handle missing values
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        self.handle_missing_values(missing_cols)
        
        # Create vehicle age feature
        self.create_vehicle_age()
        
        # Encode high-cardinality features
        self.encode_high_cardinality()
        
        # Frequency encoding for location features
        location_cols = ['MainCrestaZone', 'SubCrestaZone']
        self.frequency_encode(location_cols)
        
        # Label encoding for low-cardinality features
        categorical_cols = ['MaritalStatus', 'Language', 'AccountType']
        self.label_encode(categorical_cols)
        
        # Log-transform skewed features
        skewed_cols = ['TotalClaims', 'TotalPremium']
        self.log_transform(skewed_cols)
        
        # Create interaction features
        self.create_interaction_features()
        
        # Binary encoding for boolean features
        binary_cols = ['NewVehicle', 'TrackingDevice', 'WrittenOff', 'Rebuilt', 'Converted']
        self.binary_encode(binary_cols)
        
        print("Feature engineering complete.")
        return self.df

# Example usage:
# Load your dataset
# df = pd.read_csv("your_dataset.csv")

# Initialize FeatureEngineering and preprocess the data
# cols_to_drop = ['UnderwrittenCoverID', 'PolicyID', 'TransactionMonth', 'PostalCode', 'Title', 'Bank']
# feature_engineer = FeatureEngineering(df, target_variable="TotalClaims")
# processed_df = feature_engineer.process(cols_to_drop=cols_to_drop)
