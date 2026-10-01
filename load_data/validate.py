import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Validations:

    def __init__(self, df, profile):
        self.df = df
        self.profile = profile
        self.target = None
        self.problem_type = None

    # --------------------------------------------------
    # STEP 1: Validate Dataset
    # --------------------------------------------------

    def validate(self):

        validation_results = {}
        print(self.df.columns)
        # Row validation
        validation_results["row_count"] = (
            self.df.shape[0] == self.profile["dataset"]["rows"]
        )

        # Column validation
        validation_results["column_count"] = (
            self.df.shape[1] == self.profile["dataset"]["columns"]
        )

        # Missing value validation
        profile_missing = pd.Series(
            self.profile["missing_values"]
        )

        current_missing = self.df.isnull().sum()

        validation_results["missing_values"] = (
            current_missing.equals(profile_missing)
        )

        # Unique value validation
        profile_unique = pd.Series(
            self.profile["unique_values"]
        )

        current_unique = self.df.nunique()

        validation_results["unique_values"] = (
            current_unique.equals(profile_unique)
        )

        # Data type validation
        profile_dtypes = pd.Series(
            self.profile["dtypes"]
        )

        current_dtypes = self.df.dtypes.astype(str)

        validation_results["dtypes"] = (
            current_dtypes.equals(profile_dtypes)
        )

        return validation_results

    # --------------------------------------------------
    # STEP 2: Choose Target Column
    # --------------------------------------------------

    def choose_target_column(self, target_column):
        
        if target_column not in self.df.columns:
            raise ValueError(
                f"Column '{target_column}' not found in dataset."
            )

        self.target = target_column

        return self.target

    # --------------------------------------------------
    # STEP 3: Detect Problem Type
    # --------------------------------------------------

    def detect_problem_type(self):

        if self.target is None:
            raise ValueError(
                "Please choose target column first."
            )

        target_data = self.df[self.target]

        # Categorical target
        if (
            target_data.dtype == "object"
            or str(target_data.dtype) == "category"
            or target_data.dtype == "bool"
        ):
            self.problem_type = "classification"

        # Numerical target
        elif pd.api.types.is_numeric_dtype(target_data):

            unique_values = target_data.nunique()

            if unique_values <= 15:
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"

        else:
            raise ValueError(
                "Unable to determine problem type."
            )

        return self.problem_type

    # --------------------------------------------------
    # STEP 4: Split Dataset
    # --------------------------------------------------

    def split_data(self, test_size=0.2):

        if self.target is None:
            raise ValueError(
                "Please choose target column first."
            )

        if self.problem_type is None:
            self.detect_problem_type()

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Stratification for classification
        stratify = y if self.problem_type == "classification" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify
        )

        return X_train, X_test, y_train, y_test

    # --------------------------------------------------
    # STEP 5: Build Preprocessing Pipeline
    # --------------------------------------------------

    def build_preprocessor(self, X_train):

        numerical_columns = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X_train.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # Numerical preprocessing
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # Categorical preprocessing
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        # Combine both pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_columns
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns
                )
            ],
            remainder="drop"
        )

        return preprocessor