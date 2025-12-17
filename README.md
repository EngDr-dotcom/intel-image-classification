# Intel Image Classification

Image classification project using:
- CNN Baseline
- Transfer Learning (VGG-based)

## Features
- Image upload & prediction
- Model comparison
- Confidence visualization
- Prediction history
- Streamlit web interface

## Project Structure

intel-image-classification/
│
├── models/
│ cnn_baseline.h5
│ transfer_learning_model.h5
│
├── app/
│ streamlit_app.py
│
├── notebooks/
│ cnn_baseline.ipynb
│ transfer_learning.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore

## System Requirements & Setup Notes

To ensure the project runs correctly without errors, please follow the requirements and notes below.

### Python Version
- Python **3.9 or 3.10** is recommended.
- Python 3.11+ is not recommended due to potential compatibility issues with TensorFlow.

### Required Libraries
All required libraries are listed in `requirements.txt`.  
Main dependencies include:
- tensorflow
- streamlit
- numpy
- pillow
- matplotlib



## How to Run

```bash
pip install -r requirements.txt
cd app
streamlit run streamlit_app.py


