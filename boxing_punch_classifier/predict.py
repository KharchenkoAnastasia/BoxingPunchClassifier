import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features
from data_processing import (
    butterworth_filter,
    replace_anomalies_with_mean
)


def load_model(model_path: Path) -> keras.Model:
    """
    Load the trained model from disk.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return keras.models.load_model(str(model_path))


def preprocess_data(df: pd.DataFrame, cutoff_freq: float, sampling_freq: float) -> pd.DataFrame:
    """
    Preprocess the input data using the same steps as training.
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

    return df


def prepare_features(df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """
    Extract and prepare features for prediction.
    """
    # Extract features
    features_df = extract_features(df)

    # Select only the features used in training
    if selected_features:
        features_df = features_df[selected_features]

    # Standardize features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features_df)

    return pd.DataFrame(standardized_features, columns=features_df.columns)


def predict_activity(model: keras.Model, features: pd.DataFrame) -> str:
    """
    Make a prediction using the loaded model.
    """
    # Define the label mapping (should match training)
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
    reverse_mapping = {v: k for k, v in LABEL_MAPPING.items()}

    # Make prediction
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Convert to label
    predicted_label = reverse_mapping[predicted_class]
    confidence = np.max(prediction) * 100

    return predicted_label, confidence


def main():
    # Configuration
    MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.h5"
    CUTOFF_FREQUENCY = 4.0
    SAMPLING_FREQUENCY = 100.0

    # Top 10 features used in training (should match the features used in training)
    SELECTED_FEATURES = [
        'mean_accX', 'mean_accY', 'mean_accZ',
        'std_accX', 'std_accY', 'std_accZ',
        'mean_gyrX', 'mean_gyrY', 'mean_gyrZ',
        'std_gyrX'
    ]

    try:
        # Load the model
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")

        # Example: Load new data for prediction
        # Replace this with your actual data loading logic
        test_data_path = Path(__file__).parent.parent / "data" / "test_sample.csv"
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data file not found at {test_data_path}")

        # Load and preprocess the data
        new_data = pd.read_csv(test_data_path)
        processed_data = preprocess_data(new_data, CUTOFF_FREQUENCY, SAMPLING_FREQUENCY)

        # Prepare features
        features = prepare_features(processed_data, SELECTED_FEATURES)

        # Make prediction
        predicted_label, confidence = predict_activity(model, features)

        print(f"\nPrediction Results:")
        print(f"Predicted Activity: {predicted_label}")
        print(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()