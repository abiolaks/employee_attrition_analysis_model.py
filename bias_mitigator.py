from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BiasMitigator:
    def __init__(self):
        self.smote = SMOTE(random_state=42)

    def balance_data(self, X, y):
        """Balance the dataset using SMOTE"""
        X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res

    def check_fairness(self, model, X_test, y_test, sensitive_features):
        """Check model fairness across sensitive features"""
        fairness_report = {}

        # Get predictions
        y_pred = model.predict(X_test)

        for feature in sensitive_features:
            # Get unique values of the sensitive feature
            unique_values = X_test[feature].unique()

            # Calculate metrics for each group
            group_metrics = {}
            for value in unique_values:
                mask = X_test[feature] == value
                group_y_test = y_test[mask]
                group_y_pred = y_pred[mask]

                if len(group_y_test) > 0:
                    group_metrics[value] = {
                        "accuracy": accuracy_score(group_y_test, group_y_pred),
                        "precision": precision_score(group_y_test, group_y_pred),
                        "recall": recall_score(group_y_test, group_y_pred),
                        "f1": f1_score(group_y_test, group_y_pred),
                        "sample_size": len(group_y_test),
                    }

            fairness_report[feature] = group_metrics

        return fairness_report
