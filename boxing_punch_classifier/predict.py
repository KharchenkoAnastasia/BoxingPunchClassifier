import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from extract_features import extract_features
from data_processing import butterworth_filter, replace_anomalies_with_mean


def load_and_preprocess_data(file_path: str, cutoff_freq: float = 4.0, sampling_freq: float = 100.0) -> pd.DataFrame:
    """
    Load and preprocess the sensor data file.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Add temporary label and hand columns for compatibility with preprocessing
    df['label'] = 'Unknown'
    df['hand'] = 'right'  # default value, will be removed later

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

    return df


def predict_movements(data_path: Path, model_path: Path) -> list:
    """
    Predict boxing movements from sensor data.
    """
    # Selected features used in training
    SELECTED_FEATURES = [
        'x_acc_mean', 'y_acc_mean', 'z_acc_mean',
        'x_gyr_mean', 'y_gyr_mean', 'z_gyr_mean',
        'x_acc_std', 'y_acc_std', 'z_acc_std',
        'x_gyr_std'
    ]

    # Movement label mapping
    LABEL_MAPPING = {
        0: 'hook right',
        1: 'hook left',
        2: 'uppercut left',
        3: 'uppercut right',
        4: 'jab left',
        5: 'jab right',
        6: 'NoActivity left',
        7: 'NoActivity right'
    }

    # Load and preprocess the data
    df = load_and_preprocess_data(str(data_path))

    try:
        # Extract features
        feature_df = extract_features(df)

        # Select only the features used in training
        feature_df = feature_df[SELECTED_FEATURES]

        # Load the model
        model = keras.models.load_model(str(model_path))

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_df)

        # Make predictions
        predictions = model.predict(features_scaled)
        predicted_classes = np.argmax(predictions, axis=1)

        # Convert predictions to labels
        predicted_labels = [LABEL_MAPPING[pred] for pred in predicted_classes]

        # Add confidence scores
        confidence_scores = np.max(predictions, axis=1)

        # Combine predictions with confidence scores
        results = list(zip(predicted_labels, confidence_scores))

        return results

    except Exception as e:
        print(f"Error during feature extraction or prediction: {e}")
        raise


def main():
    # Configuration
    ROOT_DIR = Path(__file__).parent.parent / "data"
    MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.h5"
    DATA_PATH = ROOT_DIR / "test_sample.csv"

    try:
        # Make predictions
        predictions = predict_movements(DATA_PATH, MODEL_PATH)

        # Print results
        print("\nPredicted Boxing Movements:")
        print("---------------------------")
        for i, (movement, confidence) in enumerate(predictions, 1):
            print(f"Movement {i}:")
            print(f"  Type: {movement}")
            print(f"  Confidence: {confidence:.2%}\n")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required files are in the correct location:")
        print(f"Data path: {DATA_PATH}")
        print(f"Model path: {MODEL_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Please check if your input data has all required columns: timestamp, seconds_elapsed, accZ, accY, accX, gyrZ, gyrY, gyrX")


if __name__ == "__main__":
    main()