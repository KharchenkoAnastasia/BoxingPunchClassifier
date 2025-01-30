# Use official Python runtime as a base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Copy the project code into the container
COPY boxing_punch_classifier/ boxing_punch_classifier/
COPY data/ data/
COPY model/ model/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the prediction script
CMD ["python", "boxing_punch_classifier/predict.py"]