from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import pandas as pd


class AttritionModel:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(random_state=42, probability=True),
        }
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None

    def train_models(self, X_train, y_train):
        """Train multiple models and select the best one"""
        best_score = 0

        for name, model in self.models.items():
            pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("feature_selection", SelectKBest(f_classif, k=10)),
                    ("classifier", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            self.trained_models[name] = pipeline

            score = pipeline.score(X_train, y_train)

            if score > best_score:
                best_score = score
                self.best_model = pipeline
                self.best_model_name = name

        return self.trained_models

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        evaluation_results = {}

        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            evaluation_results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
            }

        return evaluation_results

    def get_feature_importance(self, model_name):
        """Get feature importance from tree-based models"""
        model = self.trained_models[model_name]

        if hasattr(model.named_steps["classifier"], "feature_importances_"):
            preprocessor = model.named_steps["preprocessor"]
            feature_names = []

            # Numerical features
            feature_names.extend(self.numerical_features)

            # Categorical features (one-hot encoded)
            if "cat" in preprocessor.named_transformers_:
                ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                cat_features = ohe.get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_features)

            selector = model.named_steps["feature_selection"]
            selected_mask = selector.get_support()
            selected_features = [f for f, m in zip(feature_names, selected_mask) if m]

            importances = model.named_steps["classifier"].feature_importances_

            return pd.DataFrame(
                {"feature": selected_features, "importance": importances}
            ).sort_values("importance", ascending=False)

        return None

    def get_best_model(self):
        """Return the best performing model"""
        return self.best_model, self.best_model_name
