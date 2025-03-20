import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        """Loads the dataset"""
        self.df = pd.read_csv(self.file_path)

    def encode_categorical(self, encoding_type="default"):
     """Encodes categorical variables using Label Encoding, Frequency, or Target Encoding."""
     for col in self.categorical_cols:
        self.df[col] = self.df[col].fillna("Unknown")  # Handle missing values
        unique_vals = self.df[col].nunique()

        if unique_vals > self.high_cardinality_threshold:  # High-cardinality features
            print(f"Encoding {col} using frequency encoding (high cardinality: {unique_vals} unique values)")
            self.df[col] = self.df[col].map(self.df[col].value_counts(normalize=True))
        
        elif encoding_type == "target" and unique_vals > 10:  # Use target encoding for high-cardinality
            print(f"Encoding {col} using target encoding")
            from category_encoders import TargetEncoder
            target_encoder = TargetEncoder(cols=[col])
            self.df[col] = target_encoder.fit_transform(self.df[col], self.df[self.target_variable])

        elif encoding_type == "one-hot" and unique_vals < 10:  # Use one-hot for small cardinality
            print(f"Encoding {col} using one-hot encoding")
            self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
        
        else:  # Default low-cardinality behavior
            print(f"Encoding {col} using Label Encoding (low cardinality: {unique_vals} unique values)")
            from sklearn.preprocessing import OrdinalEncoder
            ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.df[col] = ordinal_encoder.fit_transform(self.df[[col]])

    def scale_features(self):
        """Scales numerical features using StandardScaler"""
        scaler = StandardScaler()
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.drop(self.target_column)
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

    def split_data(self, test_size=0.2, random_state=42):
        """Splits the dataset into training and testing sets"""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess(self):
        """Runs the entire preprocessing pipeline"""
        self.load_data()
        self.encode_categorical()
        self.scale_features()
        self.split_data()

        return self.X_train, self.X_test, self.y_train, self.y_test
