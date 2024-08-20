# Sentiment Analysis System for Movie Reviews
![banner](banner.webp)

## Project Title
**Sentiment Analysis System for Movie Reviews Using Traditional Machine Learning and Self-Training**

## Description
Developed a sentiment analysis system using scikit-learn and XGBoost to accurately classify movie reviews as positive or negative, addressing class imbalance with appropriate techniques to improve customer feedback analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Technical Summary](#technical-summary)
- [Analysis](#analysis)
- [Challenges](#challenges)
- [Solutions](#solutions)
- [Conclusion](#conclusion)
- [Improvements and Future Work](#improvements-and-future-work)
- [Serialization and Streamlit App](#serialization-and-streamlit-app)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites
- Python 3.8 or higher
- Visual Studio Code (VS Code)
- Jupyter Notebook extension for VS Code
- Git

### Clone the Repository and Set Up Environment
```sh
# Clone the repository
git clone https://github.com/thehamzza/Sentiment-Analysis-of-Movie-Reviews-with-ML.git
cd sentiment-analysis-system

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt

# Download and extract the dataset
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
mkdir data
tar -xvzf aclImdb_v1.tar.gz -C data
```

## Usage

### Running the Jupyter Notebook in VS Code
1. **Open the Project in VS Code**:
   - Start VS Code.
   - Open the project folder (`sentiment-analysis-system`).

2. **Set Up Python Interpreter**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the Command Palette.
   - Type `Python: Select Interpreter` and select the virtual environment you created (`venv`).

3. **Install Jupyter Extension**:
   - If you haven't already, install the Jupyter extension for VS Code from the Extensions view (`Ctrl+Shift+X`).

4. **Open and Run the Notebook**:
   - Navigate to the `main.ipynb` file in the Explorer view.
   - Click on the file to open it.
   - Click on the `Run All` button at the top to execute all cells in the notebook, or run cells individually.

### Running the Analysis
1. Follow the instructions in the Jupyter Notebook to:
   - Load and preprocess the dataset.
   - Train the models (Logistic Regression, SVM, XGBoost).
   - Handle class imbalance.
   - Evaluate the models.
   - Perform self-training with unsupervised data.

## Technical Summary

### Data Preprocessing
1. **Loading the Dataset**: Loaded 50,000 movie reviews (25,000 train and 25,000 test) and additional 50,000 unlabeled documents for unsupervised learning.
2. **Cleaning the Data**: Removed stop words, punctuation, and performed lemmatization using NLTK.
3. **Feature Extraction**: Converted text data into numerical features using TF-IDF with 5000 features.

### Model Selection and Training
1. **Logistic Regression**: Trained using class weights to handle imbalance.
2. **Support Vector Machine (SVM)**: Trained with class weights to handle imbalance.
3. **XGBoost**: Trained with `scale_pos_weight` to handle class imbalance.

### Class Imbalance Handling
1. Used class weights to balance the training process for Logistic Regression and SVM.
2. Applied `scale_pos_weight` for XGBoost to handle class imbalance.

### Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### Testing User Input Review
We tested the models on a live user input review with ambiguous language to evaluate their performance. 
The review was: "It was just okay, I had seen better movies before. Only if they could make it more interesting." 
Despite the unclear sentiment, all three models (Logistic Regression, SVM, XGBoost) correctly predicted it as Negative (0). 
This demonstrates the models' accuracy and robustness, even with challenging and confusing inputs, ensuring reliable sentiment analysis for real-world applications.

### Results
- **Logistic Regression**
  - Accuracy: 0.87952
  - Precision: 0.8786113328012769
  - Recall: 0.88072
  - F1 Score: 0.87966440271674
- **SVM**
  - Accuracy: 0.88024
  - Precision: 0.8809714652132093
  - Recall: 0.87928
  - F1 Score: 0.8801249199231262
- **XGBoost**
  - Accuracy: 0.8544
  - Precision: 0.8426140757927301
  - Recall: 0.8716
  - F1 Score: 0.8568619740464019

## Serialization and Streamlit App

### Why Serialization?
We selected the SVM model as it outperformed others based on accuracy, precision, recall, and F1 score. To make this model accessible to end-users, we serialized (pickled) both the trained SVM model and the TF-IDF vectorizer. Serialization allows the model and vectorizer to be loaded in a production environment without retraining, enabling fast predictions.

### Streamlit App for End-to-End User Experience
We built a Streamlit web application to provide an intuitive interface for users to test the live sentiment analysis model. Users can input their own movie reviews and receive instant predictions on whether the sentiment is positive or negative. You can try the app at [movie.streamlit.app](https://movie.streamlit.app).

## Acknowledgements
- This project uses the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) provided by Stanford AI Lab.
- The project was developed using Python, scikit-learn, NLTK, and XGBoost.
- Special thanks to the authors of the dataset for making it available for research purposes.
```
