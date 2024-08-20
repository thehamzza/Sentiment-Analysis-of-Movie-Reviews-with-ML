import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# Set page configuration with title and favicon
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="favicon.ico",  # You can use an emoji or a custom image (e.g., "favicon.ico")
    layout="centered",  # Options: "centered" or "wide"
    initial_sidebar_state="auto",  # Options: "auto", "expanded", "collapsed"
)

# Load pre-trained model and vectorizer
model = joblib.load('svm_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Custom CSS for styling
st.markdown("""
    <style>
    .title-style {
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
    .emoji-style {
        font-size: 50px;
        margin-right: 10px;
    }
    .review-textarea {
        font-family: 'Arial', sans-serif;
        padding: 10px;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        width: 100%;
        height: 150px;
    }
    .analyze-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        cursor: pointer;
        margin-top: 10px;
    }
    .contact-form {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        width: 100%;
    }
    .contact-submit-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        cursor: pointer;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Top navigation using st.selectbox
page = st.selectbox("Menu", ["Home", "About", "Contact"], index=0)

# Home Page
if page == "Home":
    
    st.markdown("<div class='title-style'><h1> Sentiment Analysis App </h1></div>", unsafe_allow_html=True)
    st.write("Welcome! Enter a movie review to see if it's **positive** or **negative**. Let's get started!")

    # Display a banner image
    st.image("banner.webp", use_column_width=True)

    # Input for new review
    st.subheader("Share Your Review 📝")
    review_text = st.text_area("Type your movie review here...", height=150)

    if st.button("Analyze Sentiment"):
        if review_text:
            # Preprocess and predict sentiment
            review_preprocessed = vectorizer.transform([review_text])
            sentiment = model.predict(review_preprocessed)[0]
            sentiment_text = "Positive 😊" if sentiment == 1 else "Negative 😢"
            st.write(f"The review sentiment is: **{sentiment_text}**")
        else:
            st.error("Please enter a review text!")

# About Page
elif page == "About":
    st.markdown("<div class='title-style'><h1>About the App 🎬</h1></div>", unsafe_allow_html=True)
    
    st.header("Welcome!")
    st.write("This Sentiment Analysis App is designed to help movie review websites automatically classify user reviews as positive or negative.")

    st.header("How It Works 💡")
    st.write("""
    - **Automatic Moderation**: Filters extreme sentiments to maintain a positive community vibe.
    - **User Insights**: Shows the percentage of positive reviews, helping users make better choices.
    - **Trending Topics**: Highlights trending movies based on review sentiments.
    """)

    st.header("Why SVM? 🤖")
    st.write("We tested several models, and the Support Vector Machine (SVM) was chosen for its excellent balance of accuracy and efficiency.")

    st.header("Model Metrics Comparison 📊")
    st.write("Here's a quick comparison of model performance:")

    # Sample data for demonstration (replace with actual data)
    models = ['Logistic Regression', 'SVM', 'XGBoost']
    accuracy = [0.8795, 0.8802, 0.8544]
    precision = [0.8786, 0.881, 0.8426]
    recall = [0.8807, 0.8793, 0.8716]
    f1_score = [0.8797, 0.8801, 0.8569]

    # Plotting with matplotlib
    barWidth = 0.15
    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    plt.figure(figsize=(10, 5))
    plt.bar(r1, accuracy, color='b', width=barWidth, edgecolor='grey', label='Accuracy')
    plt.bar(r2, precision, color='g', width=barWidth, edgecolor='grey', label='Precision')
    plt.bar(r3, recall, color='r', width=barWidth, edgecolor='grey', label='Recall')
    plt.bar(r4, f1_score, color='y', width=barWidth, edgecolor='grey', label='F1 Score')

    plt.xlabel('Models', fontweight='bold')
    plt.xticks([r + 1.5*barWidth for r in range(len(models))], models)
    plt.ylabel('Scores', fontweight='bold')
    plt.ylim(0.84, 0.89)
    plt.title('Comparison of Evaluation Metrics for Different Models')
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot(plt)

    st.header("Project Code and Technical Report 💼")
    st.write("Check out the project code and technical report on [GitHub](https://github.com/your-github-profile/your-repo) for more details on data preprocessing, model training, and evaluation.")

# Contact Page
elif page == "Contact":
    st.markdown("<div class='title-style'><h1>Get in Touch 📬</h1></div>", unsafe_allow_html=True)
    st.write("""
    We'd love to hear from you! Connect with us on:
    - **Email**: [its.hamza100@gmail.com](its.hamza100@gmail.com)
    - **LinkedIn**: [linkedin.com/in/muhammad-hamza-shakoor/](https://www.linkedin.com/in/muhammad-hamza-shakoor/)
    - **GitHub**: [github.com/thehamzza](https://github.com/thehamzza)
    - **Medium**: [@the.hamza](https://medium.com/@the.hamza)
    - **Website**: [mhamza.site](https://mhamza.site)
    """)

    st.write("Or drop a message here 👇")
    with st.form(key='contact_form'):
        name = st.text_input('Name')
        message = st.text_area('Message')
        submit_button = st.form_submit_button(label='Send Message', help="Click to send your message")

        if submit_button:
            st.success("Thank you for reaching out! We will get back to you soon.")
