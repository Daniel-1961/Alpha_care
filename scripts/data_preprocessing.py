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

    def encode_categorical(self):
        """Encodes categorical variables using One-Hot Encoding"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoded_features = encoder.fit_transform(self.df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
        
        self.df = self.df.drop(columns=categorical_cols)
        self.df = pd.concat([self.df, encoded_df], axis=1)

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

    def preprocess(self):
        """Runs the entire preprocessing pipeline"""
        self.load_data()
        self.encode_categorical()
        self.scale_features()
        self.split_data()

        return self.X_train, self.X_test, self.y_train, self.y_test
