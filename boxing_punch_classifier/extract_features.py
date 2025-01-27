from scipy.signal import find_peaks
import numpy as np
import pandas as pd


def extract_features(sensor_data_df):
    x_list = []
    y_list = []
    z_list = []

    x_gyr_list = []
    y_gyr_list = []
    z_gyr_list = []

    train_labels = []

    window_size = 200
    step_size = 100

    i = 0
    while i <= sensor_data_df.shape[0] - window_size and i + window_size <= sensor_data_df.shape[0]:
        xs = sensor_data_df['accX'].values[i: i + window_size]
        ys = sensor_data_df['accY'].values[i: i + window_size]
        zs = sensor_data_df['accZ'].values[i: i + window_size]

        xg = sensor_data_df['gyrX'].values[i: i + window_size]
        yg = sensor_data_df['gyrY'].values[i: i + window_size]
        zg = sensor_data_df['gyrZ'].values[i: i + window_size]

        # Using pandas mode
        label = sensor_data_df['label'][i: i + window_size].mode()[0]

        x_list.append(xs)
        y_list.append(ys)
        z_list.append(zs)

        x_gyr_list.append(xg)
        y_gyr_list.append(yg)
        z_gyr_list.append(zg)

        train_labels.append(label)
        i = i + step_size

    # Statistical Features on raw x, y and z in the time domain
    X_train = pd.DataFrame()

    # Mean
    X_train['x_acc_mean'] = pd.Series(x_list).apply(lambda x: x.mean())
    X_train['y_acc_mean'] = pd.Series(y_list).apply(lambda x: x.mean())
    X_train['z_acc_mean'] = pd.Series(z_list).apply(lambda x: x.mean())

    X_train['x_gyr_mean'] = pd.Series(x_gyr_list).apply(lambda x: x.mean())
    X_train['y_gyr_mean'] = pd.Series(y_gyr_list).apply(lambda x: x.mean())
    X_train['z_gyr_mean'] = pd.Series(z_gyr_list).apply(lambda x: x.mean())

    # std dev
    X_train['x_acc_std'] = pd.Series(x_list).apply(lambda x: x.std())
    X_train['y_acc_std'] = pd.Series(y_list).apply(lambda x: x.std())
    X_train['z_acc_std'] = pd.Series(z_list).apply(lambda x: x.std())

    X_train['x_gyr_std'] = pd.Series(x_gyr_list).apply(lambda x: x.std())
    X_train['y_gyr_std'] = pd.Series(y_gyr_list).apply(lambda x: x.std())
    X_train['z_gyr_std'] = pd.Series(z_gyr_list).apply(lambda x: x.std())

    # avg absolute diff
    X_train['x_acc_aad'] = pd.Series(x_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['y_acc_aad'] = pd.Series(y_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['z_acc_aad'] = pd.Series(z_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    X_train['x_gyr_aad'] = pd.Series(x_gyr_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['y_gyr_aad'] = pd.Series(y_gyr_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['z_gyr_aad'] = pd.Series(z_gyr_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # min
    X_train['x_acc_min'] = pd.Series(x_list).apply(lambda x: x.min())
    X_train['y_acc_min'] = pd.Series(y_list).apply(lambda x: x.min())
    X_train['z_acc_min'] = pd.Series(z_list).apply(lambda x: x.min())

    X_train['x_gyr_min'] = pd.Series(x_gyr_list).apply(lambda x: x.min())
    X_train['y_gyr_min'] = pd.Series(y_gyr_list).apply(lambda x: x.min())
    X_train['z_gyr_min'] = pd.Series(y_gyr_list).apply(lambda x: x.min())

    # max
    X_train['x_acc_max'] = pd.Series(x_list).apply(lambda x: x.max())
    X_train['y_acc_max'] = pd.Series(y_list).apply(lambda x: x.max())
    X_train['z_acc_max'] = pd.Series(z_list).apply(lambda x: x.max())

    X_train['x_gyr_max'] = pd.Series(x_gyr_list).apply(lambda x: x.max())
    X_train['y_gyr_max'] = pd.Series(y_gyr_list).apply(lambda x: x.max())
    X_train['z_gyr_max'] = pd.Series(z_gyr_list).apply(lambda x: x.max())

    # max-min diff
    X_train['x_acc_maxmin_diff'] = X_train['x_acc_max'] - X_train['x_acc_min']
    X_train['y_acc_maxmin_diff'] = X_train['y_acc_max'] - X_train['y_acc_min']
    X_train['z_acc_maxmin_diff'] = X_train['z_acc_max'] - X_train['z_acc_min']

    X_train['x_gyr_maxmin_diff'] = X_train['x_gyr_max'] - X_train['x_gyr_min']
    X_train['y_gyr_maxmin_diff'] = X_train['y_gyr_max'] - X_train['y_gyr_min']
    X_train['z_gyr_maxmin_diff'] = X_train['z_gyr_max'] - X_train['z_gyr_min']

    # median
    X_train['x_acc_median'] = pd.Series(x_list).apply(lambda x: np.median(x))
    X_train['y_acc_median'] = pd.Series(y_list).apply(lambda x: np.median(x))
    X_train['z_acc_median'] = pd.Series(z_list).apply(lambda x: np.median(x))

    X_train['x_gyr_median'] = pd.Series(x_gyr_list).apply(lambda x: np.median(x))
    X_train['y_gyr_median'] = pd.Series(y_gyr_list).apply(lambda x: np.median(x))
    X_train['z_gyr_median'] = pd.Series(z_gyr_list).apply(lambda x: np.median(x))

    # median abs dev
    X_train['x_acc_mad'] = pd.Series(x_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['y_acc_mad'] = pd.Series(y_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['z_acc_mad'] = pd.Series(z_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    X_train['x_gyr_mad'] = pd.Series(x_gyr_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['y_gyr_mad'] = pd.Series(y_gyr_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['z_gyr_mad'] = pd.Series(z_gyr_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # interquartile range
    X_train['x_acc_IQR'] = pd.Series(x_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['y_acc_IQR'] = pd.Series(y_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['z_acc_IQR'] = pd.Series(z_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    X_train['x_gyr_IQR'] = pd.Series(x_gyr_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['y_gyr_IQR'] = pd.Series(y_gyr_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['z_gyr_IQR'] = pd.Series(z_gyr_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # number of peaks
    X_train['x_acc_peak_count'] = pd.Series(x_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['y_acc_peak_count'] = pd.Series(y_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['z_acc_peak_count'] = pd.Series(z_list).apply(lambda x: len(find_peaks(x)[0]))

    X_train['x_gyr_peak_count'] = pd.Series(x_gyr_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['y_gyr_peak_count'] = pd.Series(y_gyr_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['z_gyr_peak_count'] = pd.Series(z_gyr_list).apply(lambda x: len(find_peaks(x)[0]))

    X_train['label'] = pd.Series(train_labels)

    return X_train
