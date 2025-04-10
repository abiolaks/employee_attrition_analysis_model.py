import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.categorical_features = None
        self.numerical_features = None

    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        self.data = pd.read_csv(self.data_path)

        # Create synthetic attrition target (for demo)
        # In real scenarios, you'd have an actual attrition column
        self.data["Attrition"] = (self.data["Employee Tenure (Years)"] < 2).astype(int)

        # Separate features and target
        self.X = self.data.drop("Attrition", axis=1)
        self.y = self.data["Attrition"]

        # Identify feature types
        self.categorical_features = self.X.select_dtypes(
            include=["object", "bool"]
        ).columns.tolist()
        self.numerical_features = self.X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        # Handle special columns
        if "Trainings Offered by Company" in self.numerical_features:
            self.numerical_features.remove("Trainings Offered by Company")
            self.categorical_features.append("Trainings Offered by Company")
        if "Certifications Given by Company" in self.numerical_features:
            self.numerical_features.remove("Certifications Given by Company")
            self.categorical_features.append("Certifications Given by Company")

        # Create preprocessing pipelines
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def get_preprocessed_data(self):
        """Return preprocessed data"""
        return self.X_train, self.X_test, self.y_train, self.y_test, self.preprocessor
