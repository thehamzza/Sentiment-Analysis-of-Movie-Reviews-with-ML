```markdown
# Sentiment Analysis System for Movie Reviews

## Project Title
**Sentiment Analysis System for Movie Reviews Using Traditional Machine Learning and Self-Training**

## Description
Developed a sentiment analysis system using scikit-learn and XGBoost to accurately classify movie reviews as positive or negative, addressing class imbalance with appropriate techniques to improve customer feedback analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Technical Summary](#technical-summary)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Git

### Clone the Repository and Set Up Environment
```sh
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-system.git
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

### Running the Jupyter Notebook
1. Start Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
2. Open the `Sentiment_Analysis.ipynb` file in Jupyter Notebook.

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

### Self-Training with Unlabeled Data
1. Predicted labels for the unlabeled data using the initial model.
2. Selected the most confident predictions to create additional labeled data.
3. Retrained the model using the combined dataset of original and newly labeled data.

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

## Acknowledgements
- This project uses the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) provided by Stanford AI Lab.
- The project was developed using Python, scikit-learn, NLTK, and XGBoost.
- Special thanks to the authors of the dataset for making it available for research purposes.
```
