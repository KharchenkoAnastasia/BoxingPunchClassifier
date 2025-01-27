import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def butterworth_filter(df, accelerometer_cols, gyroscope_cols, cutoff_freq, sample_freq, order=4):
    """
    Apply a Butterworth filter to accelerometer and gyroscope data in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing sensor data.
        accelerometer_cols (list): List of accelerometer column names.
        gyroscope_cols (list): List of gyroscope column names.
        cutoff_freq (float): The cutoff frequency for the Butterworth filter (in Hz).
        sample_freq (float): The sampling frequency of the data (in Hz).
        order (int): The order of the Butterworth filter (default is 4).

    Returns:
        pd.DataFrame: A new DataFrame with filtered accelerometer and gyroscope data.
    """
    filtered_df = df.copy()  # Create a copy of the input DataFrame to store the filtered data

    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * sample_freq

    # Calculate the normalized cutoff frequency
    normalized_cutoff_freq = cutoff_freq / nyquist_freq

    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff_freq, btype='low', analog=False)

    # Apply the Butterworth filter to accelerometer columns
    for col in accelerometer_cols:
        filtered_df[col] = filtfilt(b, a, df[col])

    # Apply the Butterworth filter to gyroscope columns
    for col in gyroscope_cols:
        filtered_df[col] = filtfilt(b, a, df[col])

    return filtered_df


def detect_anomalies(data, threshold=6, plot=True):
    """
    Detect anomalies in the sensor data using Z-score.

    Parameters:
        data (pd.DataFrame): DataFrame containing sensor data.
                             It should have columns: 'timestamp', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'.
        threshold (float): Z-score threshold. Default is 3.
        plot (bool): Whether to plot the data with anomalies highlighted. Default is False.

    Returns:
        pd.DataFrame: DataFrame containing the anomalous rows.
    """

    numerical_columns = data.iloc[:, 2:-2]

    # Calculate the Z-score for each numerical column
    z_scores = np.abs((numerical_columns - numerical_columns.mean()) / numerical_columns.std())

    # Find rows where any column has a Z-score greater than the threshold
    anomaly_mask = (z_scores > threshold).any(axis=1)

    # Get the anomalies
    anomalies = data[anomaly_mask]

    return anomalies


def replace_anomalies_with_mean(df, threshold=10):
    """
    Replace anomalies in the DataFrame with the mean of non-anomalous data for each numerical column.

    Parameters:
        df (pd.DataFrame): DataFrame containing sensor data.
                           It should have columns: 'timestamp', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'.
        threshold (float): Z-score threshold for anomaly detection. Default is 3.

    Returns:
        pd.DataFrame: DataFrame with anomalies replaced by mean values.
    """
    # Work with a copy of the data to avoid modifying the original DataFrame
    data = df[:]

    # Detect anomalies in the data
    anomalies = detect_anomalies(data, threshold)

    # Replace anomalies with the mean values for each numerical column
    numerical_columns = data.iloc[:, 2:-2]  # Excluding the timestamp and packet number columns
    column_means = numerical_columns.mean()

    for col in numerical_columns.columns:
        data.loc[data[col].isin(anomalies[col]), col] = column_means[col]

    return data

# Assuming you have sensor_data_df defined somewhere
def balance_data_by_label(df, label_column):
    """
    Balance a DataFrame by a specified label column without data augmentation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_column (str): The name of the label column.

    Returns:
        pd.DataFrame: A balanced DataFrame with equal or fewer samples for each label.
    """
    # Get unique labels and their counts
    label_counts = df[label_column].value_counts()

    # Determine the minimum count among labels
    min_count = label_counts.min()

    # Create an empty DataFrame to store balanced data
    balanced_df = pd.DataFrame()

    # Iterate through each unique label
    for label in label_counts.index:
        # Sample data for each label to match the minimum count
        label_data = df[df[label_column] == label].sample(n=min_count, random_state=42)

        # Append the sampled data to the balanced DataFrame
        balanced_df = pd.concat([balanced_df, label_data])

    return balanced_df
