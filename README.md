# Vegetable Classification Project

## Overview
This project involves creating a web application to classify vegetable images using a VGG16-based Convolutional Neural Network (CNN) model, achieving 99% accuracy. The application is built using Streamlit for the front-end interface. The process includes data extraction, data preprocessing, model building with the VGG16 CNN model, and model evaluation. The objective is to classify images of vegetables into one of 15 categories: Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, and Tomato.



## Project Structure
- `VeggiesClassification_modelTraining.ipynb`: Jupyter notebook used to train the VGG16 model on the vegetable dataset.

- `veggiesClassification_model.keras`: Trained VGG16 model file.

- `streamlit_app.py`: Streamlit web application .py file.

- `requirements.txt`: List of Python packages required to run the application.

- `Examples_ss/`: Directory containing example screenshots of the web app.



## Installation
1. Clone the repository:
```bash
git clone https://github.com/MPoojithavigneswari/Vegetable-Classification-Project.git
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```



## Usage
Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
The web application will open in your default web browser. You can upload an image of a vegetable either from your local system or by providing an image URL.



## Dataset
Dataset is collected from kaggle. click [here](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) for dataset download



## Application Features
- **Image Upload:** Users can upload images of vegetables directly from their local machine.

- **URL Input:** Users can provide a URL to an image of a vegetable.

- **Top-3 Predictions:** The application displays the top-3 predicted classes along with their probabilities.

- **Responsive Design:** The application layout adjusts based on the screen size.



## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any improvements or bug fixes.
