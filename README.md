# Hybrid Human Activity Recognition (HAR) System using TCN and Autoencoders

![Project Banner](https://user-images.githubusercontent.com/2646532/147772714-02685822-6b30-4ac7-87c5-55452f1f1d1f.png)
*(Feel free to create your own banner!)*

This project implements a two-stage system for human activity recognition using the "Human Activity Recognition Using Smartphones Dataset." It combines a supervised Temporal Convolutional Network (TCN) for accurate classification of known activities with an unsupervised Autoencoder for detecting anomalous or transitional movements.

---

## Table of Contents

- [Hybrid Human Activity Recognition (HAR) System using TCN and Autoencoders](#hybrid-human-activity-recognition-har-system-using-tcn-and-autoencoders)
  - [Table of Contents](#table-of-contents)
  - [Project Goal](#project-goal)
  - [Key Features](#key-features)
  - [Tech Stack and Platforms](#tech-stack-and-platforms)
  - [Dataset](#dataset)
  - [Installation and Setup](#installation-and-setup)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Methodology](#methodology)
    - [Part 1: Anomaly Detection with an Autoencoder](#part-1-anomaly-detection-with-an-autoencoder)
    - [Part 2: Activity Classification with a TCN](#part-2-activity-classification-with-a-tcn)
    - [System Workflow](#system-workflow)
  - [Results and Evaluation](#results-and-evaluation)
  - [Future Work](#future-work)
  - [Contributors](#contributors)

---

## Project Goal

The primary goal is to build a robust HAR system that can:
1.  **Accurately classify** a set of known human activities (Walking, Sitting, etc.).
2.  **Identify and flag** activities that do not fall into the known categories, treating them as anomalies.

This hybrid approach mimics real-world applications where unexpected events (like a fall) are just as important to detect as common activities.

## Key Features

*   **High-Accuracy Classifier:** Uses a Temporal Convolutional Network (TCN) to capture complex temporal patterns in sensor data.
*   **Unsupervised Anomaly Detector:** Employs a 1D Convolutional Autoencoder to learn a representation of "normal" activities and detect deviations.
*   **Modular Design:** The anomaly detector and classifier are developed as separate components that work in a sequential pipeline.
*   **Data Visualization:** Includes Jupyter notebooks for data exploration and visualization of model performance (confusion matrices, reconstruction error plots).

## Tech Stack and Platforms

*   **Language:** Python (3.8+)
*   **Core Libraries:**
    *   **TensorFlow / Keras** for model building and training.
    *   **Scikit-learn** for data splitting, preprocessing, and performance metrics (e.g., confusion matrix).
    *   **Pandas** for data manipulation.
    *   **NumPy** for numerical operations.
    *   **Matplotlib / Seaborn** for plotting and visualization.
*   **Platform:**
    *   **Jupyter Notebooks** for experimentation and exploration.
    *   **Google Colab** is an excellent choice as it provides free GPU access for faster model training.
    *   **GitHub** for version control and collaboration.

## Dataset

This project uses the **Human Activity Recognition Using Smartphones Dataset**.

*   **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
*   **Description:** The dataset contains sensor readings (accelerometer and gyroscope) from 30 subjects performing six standard activities. The data is pre-processed into time windows of 2.56 seconds.

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [URL to your GitHub repository]
    cd [repository-name]
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    *   Run the provided script to download and extract the data:
        ```bash
        python download_data.py
        ```
    *   Or, manually download it from the UCI link above and place it in the `data/` directory.

## Usage

1.  **Explore the data:**
    *   Open and run the `1_Data_Exploration.ipynb` notebook to understand the dataset.

2.  **Train the Anomaly Detector:**
    *   Run the autoencoder training script:
        ```bash
        python train_autoencoder.py
        ```
    *   This will save the trained model to the `models/` directory and output a reconstruction error threshold.

3.  **Train the Activity Classifier:**
    *   Run the TCN training script:
        ```bash
        python train_tcn.py
        ```
    *   This will save the trained TCN model to the `models/` directory.

4.  **Run the full pipeline:**
    *   Use the `run_inference.py` script to test a sample data point through the full system.
        ```bash
        python run_inference.py --sample_id [some_id]
        ```

## Project Structure
├── data/har_using_smartphones
│ └── UCI HAR Dataset/ # Folder containing the dataset files
├── notebooks/
│ ├── 1_Data_Exploration.ipynb
│ └── 2_Model_Evaluation.ipynb
├── models/ # Saved models will be stored here
│ ├── autoencoder.h5
│ └── tcn_classifier.h5
├── src/
│ ├── data_loader.py
│ ├── autoencoder_model.py
│ └── tcn_model.py
├── train_autoencoder.py
├── train_tcn.py
├── run_inference.py
├── requirements.txt
└── README.md

## Methodology

Our system uses a two-stage process for analysis.

### Part 1: Anomaly Detection with an Autoencoder

We first train a 1D Convolutional Autoencoder exclusively on "static" activities (`SITTING`, `STANDING`, `LAYING`). The model learns to reconstruct these predictable patterns with very low error. When presented with a dynamic or unknown activity, its inability to reconstruct it accurately results in a high reconstruction error, flagging it as an anomaly.

### Part 2: Activity Classification with a TCN

Data that passes the anomaly check is fed into a Temporal Convolutional Network (TCN). The TCN uses dilated, causal convolutions to efficiently analyze the entire time-series window, capturing long-range dependencies to accurately classify the data into one of the six known activities.

### System Workflow

1.  **Input:** A new window of sensor data is received.
2.  **Anomaly Check:** The data is passed to the trained Autoencoder.
3.  **Decision:**
    *   If `reconstruction_error > threshold`, the activity is labeled **"Unknown/Anomaly"**.
    *   If `reconstruction_error <= threshold`, the data proceeds to the next stage.
4.  **Classification:** The TCN classifies the "normal" data and outputs a specific activity label (e.g., "WALKING").

## Results and Evaluation

*(This section should be filled out as you complete the project.)*

*   **Autoencoder Performance:**
    *   We achieved a clear separation in reconstruction error between static and dynamic activities.
    *   **[Insert a histogram plot of reconstruction errors here]**
*   **TCN Performance:**
    *   Our TCN model achieved an accuracy of **[XX.X]%** on the test set.
    *   **[Insert a confusion matrix plot here]**

## Future Work

*   **Real-time Implementation:** Adapt the system to work with a live stream of sensor data.
*   **Explore other Architectures:** Compare the TCN's performance with an LSTM or GRU-based classifier.
*   **UI for Demonstration:** Build a simple web interface using Streamlit or Flask to visualize the results in real-time.

## Contributors

*   Leiyese (github.com/leiyese/)
*   superPiroz (github.com/SuperPiroz)