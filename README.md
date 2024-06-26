# Sentiment Analysis System for Movie Reviews

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

## Analysis

### Logistic Regression vs. SVM
- **SVM** had slightly better performance metrics due to its ability to find an optimal separating hyperplane, leading to better generalization.
- Both models performed well, likely due to the linear separability of the data after TF-IDF transformation.

### XGBoost
- **XGBoost** requires careful tuning. The default parameters might not have been optimal for this dataset.
- Potential overfitting without proper tuning, leading to lower performance on the test data.
- Further adjustments and techniques might be needed for better handling of class imbalance.

## Challenges
1. **Preprocessing the Text Data**:
   - During preprocessing, some words were losing characters, which affected the quality of the data. This issue was particularly challenging as it could impact the model's performance.
   - Example: "movie" was becoming "movi" and "characters" was becoming "charact".
   
2. **Handling Class Imbalance**:
   - The dataset had an imbalance between positive and negative reviews, requiring techniques to ensure balanced learning.

## Solutions
1. **Text Preprocessing**:
   - Used NLTK for text preprocessing, including stop word removal, punctuation removal, and lemmatization.
   - Implemented various methods and refined preprocessing steps to preserve the integrity of the words.
   - Example: Ensured that lemmatization was done correctly and stopwords were removed without altering meaningful parts of the text.

2. **Class Imbalance**:
   - Applied class weights to the models to handle class imbalance.
   - Used `class_weight='balanced'` for Logistic Regression and SVM.
   - Set `scale_pos_weight` parameter for XGBoost.

## Conclusion
The SVM model performed the best overall in terms of accuracy, precision, and F1 score. Class imbalance was effectively handled using class weights. Logistic Regression also performed very well, while the XGBoost model had lower performance compared to the other two. The challenges faced during text preprocessing were resolved through careful refinement of preprocessing steps, ensuring high-quality data for model training. This project successfully developed a robust sentiment analysis system, leveraging labeled and unlabeled data to check performance.

## Improvements and Future Work

### Model Comparison and Optimization
To further enhance the performance of the sentiment analysis system, future work could involve comparing a broader range of machine learning models, including deep learning approaches such as LSTM or BERT, which have shown significant improvements in natural language processing tasks. Hyperparameter optimization techniques, such as grid search or random search, could be employed to fine-tune the models for better accuracy and generalization.

### Semi-Supervised and Unsupervised Learning
Utilizing the large amount of unlabeled data available in the dataset through semi-supervised learning approaches can provide significant improvements. Techniques like self-training, where an initial model is trained on labeled data and then used to predict labels for the unlabeled data, can iteratively improve the model's performance. Additionally, methods like label propagation can help spread label information from labeled to unlabeled data based on similarity. For a completely unsupervised approach, clustering algorithms or topic modeling can be explored to identify patterns and sentiments in the reviews without relying on labeled data.

### Advanced Text Preprocessing and Feature Extraction
Improving text preprocessing steps, such as handling negations more effectively and using advanced lemmatization techniques, can lead to better feature representation. Additionally, experimenting with different feature extraction methods, such as word embeddings (Word2Vec, GloVe) or contextual embeddings (BERT embeddings), can capture more semantic information from the text, potentially leading to better model performance.

By implementing these improvements and exploring various machine learning techniques, the sentiment analysis system can be made more robust, accurate, and capable of handling diverse and large-scale datasets effectively.


## Acknowledgements
- This project uses the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) provided by Stanford AI Lab.
- The project was developed using Python, scikit-learn, NLTK, and XGBoost.
- Special thanks to the authors of the dataset for making it available for research purposes.
```
