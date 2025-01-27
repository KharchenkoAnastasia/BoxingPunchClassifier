from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def feature_importance_random_forest(X_train_data_nor, Y_train, df_train):
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Random Forest classifier
    rf_classifier.fit(X_train_data_nor, Y_train['label'])

    # Get feature importances from the trained model
    feature_importances = rf_classifier.feature_importances_

    # Sort feature importances in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]

    # Get feature names from the original dataset
    feature_names = df_train.columns[:-1]  # Exclude the 'label' column

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train_data_nor.shape[1]), feature_importances[sorted_indices])
    plt.xticks(range(X_train_data_nor.shape[1]), np.array(feature_names)[sorted_indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title('Feature selection with Random Forest')
    plt.tight_layout()
    plt.show()

    return feature_names

def mutual_information_feature_selection(X_train_data_nor, Y_train):
    num_features_to_select =X_train_data_nor.shape[1]
    mi_scores = mutual_info_classif(X_train_data_nor, Y_train['label'])
    sorted_indices = np.argsort(mi_scores)[::-1]
    selected_feature_indices = sorted_indices[:num_features_to_select]
    selected_mi_scores = mi_scores[selected_feature_indices]
    selected_feature_names = X_train_data_nor.columns[selected_feature_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(selected_mi_scores)), selected_mi_scores, tick_label=selected_feature_names)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Features')
    plt.ylabel('Mutual Information Score')
    plt.title('Feature selection with Mutual Information Scores')
    plt.tight_layout()
    plt.show()

    return selected_feature_names

def recursive_feature_elimination(X_train_data_nor, Y_train):
    base_classifier = LogisticRegression()
    num_features_to_select = 5
    rfe = RFE(estimator=base_classifier, n_features_to_select=num_features_to_select)
    rfe.fit(X_train_data_nor, Y_train['label'])
    sorted_feature_indices = np.argsort(rfe.ranking_)[::-1]
    sorted_feature_names = X_train_data_nor.columns[sorted_feature_indices]
    sorted_rfe_ranking = rfe.ranking_[sorted_feature_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(sorted_rfe_ranking) + 1), sorted_rfe_ranking)
    plt.xticks(range(1, len(sorted_rfe_ranking) + 1), sorted_feature_names, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Feature Ranking')
    plt.title('Feature selection with RFE')
    plt.tight_layout()
    plt.show()

    return sorted_feature_names
