import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from extract_features import extract_features
from feature_selection import feature_importance_random_forest
from data_processing import (
    butterworth_filter,
    replace_anomalies_with_mean,
    balance_data_by_label
)


def load_experiment_data(root_dir: Path) -> pd.DataFrame:
    """
    Load and combine all experiment data files.
    """
    experiment_files = {
        'jab_left': ['slow', 'fast'],
        'jab_right': ['fast', 'slow'],
        'uppercut_right': ['slow', 'fast'],
        'uppercut_left': ['fast', 'slow'],
        'hook_right': ['slow', 'fast'],
        'hook_left': ['fast', 'slow'],
        'all': ['right', 'left']
    }

    dataframes = []
    for movement, speeds in experiment_files.items():
        for speed in speeds:
            filename = f"acc_gyr_experiment_{movement}_{speed}.csv"
            file_path = root_dir / filename
            if file_path.exists():
                dataframes.append(pd.read_csv(file_path))

    return pd.concat(dataframes, ignore_index=True)


def preprocess_data(df: pd.DataFrame, cutoff_freq: float, sampling_freq: float) -> pd.DataFrame:
    """
    Preprocess the sensor data with various cleaning and filtering steps.
    """
    # Handle missing values
    df['label'].fillna(method='ffill', inplace=True)

    # Sort and reset index
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Apply Butterworth filter
    df = butterworth_filter(
        df,
        ['accX', 'accY', 'accZ'],
        ['gyrX', 'gyrY', 'gyrZ'],
        cutoff_freq,
        sampling_freq
    )

    # Replace anomalies
    df = replace_anomalies_with_mean(df, 3)

    # Combine label and hand columns
    df['label'] = df.apply(lambda row: f"{row['label']} {row['hand']}", axis=1)
    df = df.drop(columns=['hand'])

    return df


def create_model(input_shape: int, num_classes: int) -> keras.Model:
    """
    Create and compile the neural network model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='Adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history):
    """
    Plot the training and validation accuracy over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot the confusion matrix as a heatmap.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def main():
    # Configuration
    ROOT_DIR = Path(__file__).parent.parent / "data"
    CUTOFF_FREQUENCY = 4.0
    SAMPLING_FREQUENCY = 100.0
    LABEL_MAPPING = {
        'hook right': 0,
        'hook left': 1,
        'uppercut left': 2,
        'uppercut right': 3,
        'jab left': 4,
        'jab right': 5,
        'NoActivity left': 6,
        'NoActivity right': 7
    }

    # Load and preprocess data
    sensor_data_df = load_experiment_data(ROOT_DIR)
    processed_df = preprocess_data(sensor_data_df, CUTOFF_FREQUENCY, SAMPLING_FREQUENCY)

    # Balance dataset
    balanced_df = balance_data_by_label(processed_df, 'label')

    # Extract features
    feature_df = extract_features(balanced_df)

    # Split data
    df_train, df_test = train_test_split(feature_df, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(df_train.iloc[:, :-1]),
        columns=df_train.columns[:-1]
    )
    X_test = pd.DataFrame(
        scaler.transform(df_test.iloc[:, :-1]),
        columns=df_test.columns[:-1]
    )

    # Prepare labels
    Y_train = pd.DataFrame(df_train['label'].map(LABEL_MAPPING), columns=['label'])
    Y_test = pd.DataFrame(df_test['label'].map(LABEL_MAPPING), columns=['label'])

    # Feature selection
    selected_features = feature_importance_random_forest(X_train, Y_train, df_train)
    X_train = X_train[selected_features[:10]]
    X_test = X_test[selected_features[:10]]
    print(X_train.columns)
    # Create and train model
    model = create_model(X_train.shape[1], len(LABEL_MAPPING))
    history = model.fit(
        X_train,
        Y_train,
        epochs=10,
        validation_split=0.2,
        batch_size=32
    )

    # Save model
    model_path = Path(__file__).resolve().parent.parent / "model" / "model.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    # Generate predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Convert predictions back to labels
    reverse_mapping = {v: k for k, v in LABEL_MAPPING.items()}
    y_test_labels = np.vectorize(reverse_mapping.get)(Y_test['label'])
    y_pred_labels = np.vectorize(reverse_mapping.get)(y_pred)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(
        y_test_labels,
        y_pred_labels,
        list(LABEL_MAPPING.keys())
    )


if __name__ == "__main__":
    main()