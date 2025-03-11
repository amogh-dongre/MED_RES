import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Approach 1: One-vs-Rest Isolation Forest
def one_vs_rest_isolation_forest(X_train, y_train, X_test, y_test, random_state=42):
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    # Store predictions and anomaly scores for each class
    test_predictions = np.zeros((X_test.shape[0], n_classes))
    test_scores = np.zeros((X_test.shape[0], n_classes))
    
    # Train one model per class (treating each class as normal and others as anomalies)
    models = {}
    for i, class_label in enumerate(classes):
        # Select samples of current class
        X_train_class = X_train[y_train == class_label]
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=1000,
            max_samples='auto',
            contamination='auto',
            max_features=1,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state
        )
        
        iso_forest.fit(X_train_class)
        
        # Predict and get anomaly scores (-1 for outliers, 1 for inliers)
        predictions = iso_forest.predict(X_test)
        scores = iso_forest.decision_function(X_test)
        
        # Store the results
        test_predictions[:, i] = predictions
        test_scores[:, i] = scores
        models[class_label] = iso_forest
    
    # Convert predictions to class labels
    # Higher score means more likely to belong to that class
    y_pred = classes[np.argmax(test_scores, axis=1)]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'models': models,
        'predictions': y_pred,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'raw_scores': test_scores
    }


# Approach 2: Isolation Forest for Anomaly Detection + Classification
def isolation_forest_anomaly_plus_classifier(X_train, y_train, X_test, y_test, random_state=42):
    # Train a standard Isolation Forest for anomaly detection
    iso_forest = IsolationForest(
        n_estimators=1000,
        max_samples='auto',
        contamination=0.1,  # Adjust based on expected anomaly proportion
        max_features=1,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state
    )
    
    # Fit on all training data
    iso_forest.fit(X_train)
    
    # Get anomaly scores for training and test data
    train_scores = iso_forest.decision_function(X_train)
    test_scores = iso_forest.decision_function(X_test)
    
    # Convert scores to features for classification
    X_train_with_scores = np.column_stack([X_train, train_scores])
    X_test_with_scores = np.column_stack([X_test, test_scores])
    
    # Use a Random Forest classifier with the new feature
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        criterion='entropy', 
        max_features=1, 
        min_samples_leaf=1, 
        min_samples_split=2,
        max_depth=200, 
        n_estimators=1000, 
        random_state=random_state
    )
    
    # Train the classifier
    clf.fit(X_train_with_scores, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_with_scores)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'isolation_forest': iso_forest,
        'classifier': clf,
        'predictions': y_pred,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }


# Example usage:
def main():
    # Assuming you already have X_train, y_train, X_test, y_test
    
    # If you need to split your data:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features (recommended for Isolation Forest)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare both approaches
    print("Approach 1: One-vs-Rest Isolation Forest")
    result1 = one_vs_rest_isolation_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Accuracy: {result1['accuracy']:.4f}")
    print(result1['classification_report'])
    
    print("\nApproach 2: Isolation Forest + Classifier")
    result2 = isolation_forest_anomaly_plus_classifier(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Accuracy: {result2['accuracy']:.4f}")
    print(result2['classification_report'])
    
    # Compare with your Random Forest classifier
    from sklearn.ensemble import RandomForestClassifier
    clf_RF = RandomForestClassifier(
        criterion='entropy', 
        max_features=1, 
        min_samples_leaf=1, 
        min_samples_split=2,
        max_depth=200, 
        n_estimators=1000, 
        random_state=42
    )
    best_clf = clf_RF.fit(X_train_scaled, y_train)
    y_pred_rf = best_clf.predict(X_test_scaled)
    
    print("\nOriginal Random Forest Classifier")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf))
    
    # Visualize results with confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # One-vs-Rest Isolation Forest
    axes[0].set_title("One-vs-Rest Isolation Forest")
    cm1 = confusion_matrix(y_test, result1['predictions'])
    im1 = axes[0].imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticks(np.arange(len(np.unique(y_test))))
    axes[0].set_yticks(np.arange(len(np.unique(y_test))))
    axes[0].set_xticklabels(['Normal', 'Ictal', 'Interictal'])
    axes[0].set_yticklabels(['Normal', 'Ictal', 'Interictal'])
    
    # Isolation Forest + Classifier
    axes[1].set_title("Isolation Forest + Classifier")
    cm2 = confusion_matrix(y_test, result2['predictions'])
    im2 = axes[1].imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_xlabel('Predicted')
    axes[1].set_xticks(np.arange(len(np.unique(y_test))))
    axes[1].set_yticks(np.arange(len(np.unique(y_test))))
    axes[1].set_xticklabels(['Normal', 'Ictal', 'Interictal'])
    axes[1].set_yticklabels(['Normal', 'Ictal', 'Interictal'])
    
    # Random Forest
    axes[2].set_title("Random Forest Classifier")
    cm3 = confusion_matrix(y_test, y_pred_rf)
    im3 = axes[2].imshow(cm3, interpolation='nearest', cmap=plt.cm.Blues)
    axes[2].set_xlabel('Predicted')
    axes[2].set_xticks(np.arange(len(np.unique(y_test))))
    axes[2].set_yticks(np.arange(len(np.unique(y_test))))
    axes[2].set_xticklabels(['Normal', 'Ictal', 'Interictal'])
    axes[2].set_yticklabels(['Normal', 'Ictal', 'Interictal'])
    
    plt.tight_layout()
    plt.colorbar(im1, ax=axes.ravel().tolist())
    plt.show()

if __name__ == "__main__":
    main()