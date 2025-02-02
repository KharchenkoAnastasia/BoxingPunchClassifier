# Classification of Types of Punches in Boxing

## Statement of the Problem
The goal of this project is to classify types of punches in boxing. The classification involves three types of punches (jab, uppercut, hook) for both the left and right hands.

---

## Data Collection
Data was collected using the **Sensor Logger** add-on. This tool can be utilized for both initial analysis and commercial purposes.

### Sensor Specifications
- **Sensors used**: Accelerometer and Gyroscope
- **Vibration Frequency**: 100Hz

### Types of Punches
- Jab
- Uppercut
- Hook

---

## Experimental Design

### List of Experiments
For each type of punch, the following experiments were conducted:

1. **Jab**
   - 15 quick punches with the right hand.
   - 15 full blows with the right hand.
   - 15 quick punches with the left hand.
   - 15 full blows with the left hand.

2. **Uppercut**
   - 15 quick punches with the right hand.
   - 15 full blows with the right hand.
   - 15 quick punches with the left hand.
   - 15 full blows with the left hand.

3. **Hook**
   - 15 quick punches with the right hand.
   - 15 full blows with the right hand.
   - 15 quick punches with the left hand.
   - 15 full blows with the left hand.

4. **Mixed Punches**
   - One jab, one uppercut, and one hook with the right hand, repeated twice (average-quality execution).
   - One jab, one uppercut, and one hook with the left hand, repeated twice (average-quality execution).

### Experimental Setup
- During each punch, the phone was positioned directly on the hand, screen facing the fingers.
- A recovery pause of 2-3 seconds was allowed between different activities.

### Total Duration
- Total duration of all physical activities combined: **675 seconds**.

---

## Notes
- Data collected provides comprehensive information about the acceleration and angular velocity for various types of punches.
- This dataset can be used to develop and train machine learning models for punch classification.

---

## Future Scope
This project lays the foundation for:
- Real-time punch classification using wearable devices.
- Improved training and analysis for boxers.
- Commercial applications in sports technology.



### **Project Structure**
```
BoxingPunchClassifier/
├── boxing_punch_classifier/    # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── data_processing.py     # Data preprocessing functions
│   ├── extract_features.py    # Feature extraction logic
│   ├── feature_selection.py   # Feature selection algorithms
│   ├── predict.py             # Prediction script
│   └── train.py               # Model training script
├── data/                      # Data directory for sensor readings
├── model/                     # Directory for saved models
│   └── model.h5              # Trained model file
├── notebooks/                # Jupyter notebooks for analysis
├── Dockerfile                # Docker configuration
├── pyproject.toml            # Project build configuration
├── README.md                 # Project documentation
└── requirements.txt          # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KharchenkoAnastasia/BoxingPunchClassifier.git
cd BoxingPunchClassifier
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train a new model:

```bash
python boxing_punch_classifier/train.py

```

The training script will:
1. Load and preprocess the sensor data
2. Extract relevant features
3. Train the neural network model
4. Save the trained model to `model/model.h5`

### Making Predictions

Running with Docker

Prerequisites

Docker installed on your system

Your test data file (CSV format) in the data directory

Building the Docker Image
```bash
# Build the Docker image
docker build -t boxing-classifier .
```

Running Predictions

Basic run (using default test file):

```bash
docker run -it boxing-classifier
```


## Module Descriptions

- `data_processing.py`: Contains functions for cleaning and preprocessing sensor data, including filters and anomaly detection
- `extract_features.py`: Implements feature extraction from raw sensor data
- `feature_selection.py`: Contains algorithms for selecting the most relevant features
- `train.py`: Handles model training, evaluation, and saving
- `predict.py`: Provides interface for making predictions with trained model

## Data Format

Input data should be CSV files with the following columns:
- timestamp
- accX, accY, accZ (accelerometer readings)
- gyrX, gyrY, gyrZ (gyroscope readings)
- label (punch type)
- hand (left/right)

## Model Architecture

The classifier uses a neural network with:
- Input layer matching feature dimensions
- Two hidden layers (64 and 32 units) with ReLU activation
- Output layer with softmax activation for classification




