
```markdown
# PCOS Detection Tool

PCOS Detection Tool is a deep learning-based application designed to detect Polycystic Ovary Syndrome (PCOS) from medical images. This lightweight tool includes a trained deep learning model, a Flask backend for API-based predictions, and a simple HTML interface for user interaction.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Requirements](#dataset-requirements)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Flask App](#running-the-flask-app)
- [File Structure](#file-structure)
- [Flask API](#flask-api)
- [Frontend](#frontend)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This repository contains the code and assets required for the PCOS Detection Tool, which includes:
- **Training Model:** A CNN built with TensorFlow/Keras for image classification.
- **Flask Backend (`app.py`):** Loads the trained model and provides an API endpoint for image predictions.
- **HTML Frontend (`index.html`):** A user-friendly interface that allows users to upload images and view classification results.
- **Dataset:** A dataset of medical images organized into class-specific directories.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/pcos-detection-tool.git
   cd pcos-detection-tool
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Example `requirements.txt`:*
   ```
   tensorflow
   keras
   flask
   pillow
   numpy
   opencv-python
   ```

## Dataset Requirements

To train or retrain the model, you must have a properly structured dataset of medical images:

- **Directory Structure:**  
  Create a main dataset folder (e.g., `dataset_extracted/PCOS`) with two subdirectories:
  - `Infected` — Contains images showing signs of PCOS.
  - `Not Infected` — Contains images without PCOS.
  
- **Image Formats:**  
  Supported image formats include JPEG, PNG, and JPG.

- **Preprocessing:**  
  Images will be automatically resized to 128x128 pixels and normalized to have pixel values between 0 and 1.

Ensure your dataset follows this structure before running the training script.

## Usage

### Training the Model

If you plan to train the model on your own dataset:

1. Organize your dataset as described above.
2. Run the training script (e.g., `your_model_script.py`):

   ```bash
   python your_model_script.py
   ```

This script will:
- Load and preprocess images.
- Split data into training and testing sets.
- Train the CNN using early stopping and model checkpoint callbacks.
- Generate evaluation plots (accuracy, loss, ROC, and Precision-Recall curves).
- Save the trained model as `model.h5` (or similar).

### Running the Flask App

Once the model is trained (or if you have a pre-trained model):

1. Ensure the model file (e.g., `model.h5`) is in the project directory.
2. Start the Flask application:

   ```bash
   python app.py
   ```

3. Open your browser and navigate to `http://127.0.0.1:5000/` to access the tool.

### Uploading an Image

- Use the web interface provided in `index.html` to upload an image.
- Click the **Predict** button to receive the classification result (either "Infected" or "Not Infected").

## File Structure

```
pcos-detection-tool/
├── app.py               # Flask backend application
├── your_model_script.py # Script for training and evaluating the CNN model
├── model.h5             # Trained deep learning model
├── templates/
│   └── index.html       # Frontend user interface
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation (this file)
```

## Flask API

- **Endpoint:** `POST /predict`
- **Input:** An image file (JPEG/PNG)
- **Output:** JSON response with a classification result ("Infected" or "Not Infected")

## Frontend

The HTML interface (`index.html`) provides:
- **Image Upload:** A drag-and-drop/file selection area with image preview.
- **Result Display:** Shows the prediction result dynamically.
- **Dark Mode Toggle:** Allows switching between light and dark themes.

## Dependencies

This project uses the following libraries:
- **TensorFlow/Keras:** For building and running the deep learning model.
- **Flask:** For creating the backend API.
- **OpenCV and Pillow:** For image processing.
- **NumPy:** For numerical computations.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or request features via the issue tracker.
- Fork the repository and submit pull requests for improvements.
- Update documentation and code for clarity.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact me **Anish Sharma** at [sharmaanish310@gmail.com](mailto:sharmaanish310@gmail.com).
