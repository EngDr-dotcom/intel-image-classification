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
│ ├── cnn_baseline.h5
│ └── transfer_learning_model.h5
│
├── app/
│ └── streamlit_app.py
│
├── notebooks/
│ ├── cnn_baseline.ipynb
│ └── transfer_learning.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore


## How to Run

```bash
pip install -r requirements.txt
cd app
streamlit run streamlit_app.py
