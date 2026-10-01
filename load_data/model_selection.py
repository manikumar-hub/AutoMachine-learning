from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.svm import SVC, SVR

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


class CreateModel:

    def __init__(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.results = {}

    # -------------------------
    # CLASSIFICATION
    # -------------------------
    def train_classification(self):

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),

            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

            "DecisionTreeClassifier": DecisionTreeClassifier(
                random_state=42
            ),

            "SVC": SVC(
                probability=True,
                random_state=42
            ),

            "HistGradientBoostingClassifier":
                HistGradientBoostingClassifier(
                    random_state=42
                )
        }

        for name, model in models.items():

            print(f"\nTraining {name}...")

            # Train
            model.fit(
                self.X_train,
                self.y_train
            )

            # Prediction
            pred = model.predict(
                self.X_test
            )

            # Metrics
            report = classification_report(
                self.y_test,
                pred,
                output_dict=True
            )

            accuracy = report["accuracy"]

            precision = report["weighted avg"]["precision"]

            recall = report["weighted avg"]["recall"]

            f1 = report["weighted avg"]["f1-score"]

            # Confusion Matrix
            cm = confusion_matrix(
                self.y_test,
                pred
            )

            # ROC-AUC
            roc_auc = None

            if hasattr(model, "predict_proba"):

                try:

                    probabilities = model.predict_proba(
                        self.X_test
                    )

                    # Binary classification
                    if probabilities.shape[1] == 2:

                        roc_auc = roc_auc_score(
                            self.y_test,
                            probabilities[:, 1]
                        )

                except Exception:
                    roc_auc = None

            # Store results
            self.results[name] = {

                "model": model,

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1_score": f1,

                "roc_auc": roc_auc,

                "confusion_matrix": cm.tolist()
            }

            print(
                f"Accuracy: {accuracy:.4f}"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall: {recall:.4f}"
            )

            print(
                f"F1 Score: {f1:.4f}"
            )

            print(
                f"ROC-AUC: {roc_auc}"
            )

        return self.results


    # -------------------------
    # REGRESSION
    # -------------------------
    def train_regression(self):

        models = {

            "LinearRegression":
                LinearRegression(),

            "RandomForestRegressor":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                ),

            "DecisionTreeRegressor":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "SVR":
                SVR(),

            "Ridge":
                Ridge(),

            "Lasso":
                Lasso(),

            "HistGradientBoostingRegressor":
                HistGradientBoostingRegressor(
                    random_state=42
                )
        }

        for name, model in models.items():

            print(f"\nTraining {name}...")

            # Train
            model.fit(
                self.X_train,
                self.y_train
            )

            # Prediction
            pred = model.predict(
                self.X_test
            )

            # Metrics
            mse = mean_squared_error(
                self.y_test,
                pred
            )

            rmse = mse ** 0.5

            mae = mean_absolute_error(
                self.y_test,
                pred
            )

            r2 = r2_score(
                self.y_test,
                pred
            )

            self.results[name] = {

                "model": model,

                "mse": mse,

                "rmse": rmse,

                "mae": mae,

                "r2_score": r2
            }

            print(
                f"MSE: {mse:.4f}"
            )

            print(
                f"RMSE: {rmse:.4f}"
            )

            print(
                f"MAE: {mae:.4f}"
            )

            print(
                f"R2 Score: {r2:.4f}"
            )

        return self.results


    # -------------------------
    # BEST MODEL
    # -------------------------
    def get_best_classification_model(self):

        best_name = max(
            self.results,
            key=lambda name: self.results[name]["f1_score"]
        )

        best_model = self.results[best_name]["model"]

        return best_name, best_model


    def get_best_regression_model(self):

        best_name = max(
            self.results,
            key=lambda name: self.results[name]["r2_score"]
        )

        best_model = self.results[best_name]["model"]

        return best_name, best_model