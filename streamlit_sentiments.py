import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import string
import tensorflow as tf
from tensorflow import keras
import keras_nlp
from transformers import BertTokenizer, TFBertForSequenceClassification

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Load the exported data
df1 = pd.read_csv('exported_sentiments.csv')

# Encode the target labels
df1['Sentiments'] = df1['Sentiments'].replace({
    'negative': 0,
    'positive': 1
})
df1['Sentiments'].value_counts()

X = df1['Feedback']
y = df1['Sentiments']

# Text Preprocessing of the texts column using NLTK
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r'\b[0-9]+\b\s*', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

X_preprocessed = [preprocess_text(text) for text in X]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(pd.Series(X_preprocessed), y, test_size=0.25)

# Convert labels to one-hot encoded format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2, dtype='float32')

# Load the pretrained BERT model
model_name = "bert_tiny_en_uncased_sst2"
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    model_name,
    sequence_length=128,
)

training_data = tf.data.Dataset.from_tensor_slices(([X_train], [y_train]))
validation_data = tf.data.Dataset.from_tensor_slices(([X_test], [y_test]))

train_cached = (
    training_data.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)
test_cached = (
    validation_data.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)

# Pretrained classifier.
classifier = keras_nlp.models.BertClassifier.from_preset(
    model_name,
    preprocessor=None,
    num_classes=2,
    load_weights = True,
    activation='sigmoid'
)

classifier.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    jit_compile=True,
     metrics=["accuracy"],
)
classifier.fit(train_cached, validation_data=test_cached,epochs=10)

# Get the current working directory
# cwd = os.getcwd()
# saved_model_dir = os.path.join(cwd, "keras2")

# @st.cache
# def load_model():
#     return tf.keras.models.load_model(saved_model_dir)
# model = load_model()

# save model
classifier2.save("keras2", save_format='keras')

# load model
model = keras.models.load_model('keras2')

# Create a Streamlit app
st.set_page_config(
    page_title="Raqib Sentiments App",
    page_icon="R.png",
    layout="wide",
    # background_color="#F5F5F5"
)

# Center-align subheading and image using HTML <div> tags
st.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
        <h2>Sentiment Analysis App</h2>

    </div>
    """,
    unsafe_allow_html=True
)
st.image("sentim.jpg")

# # Add the subheading and image in a container
# with st.container():
#     st.subheader("Sentiment Analysis App")
#     st.image("sentim.jpg")

# Add an introductory paragraph
st.markdown("""
This web application is a sentiment analysis tool developed by Raqib (popularly known as raqibcodes). It can be used to determine whether user-entered text has a Positive or Negative sentiment. The underlying text classification model was trained on feedback data collected from 300 level undergraduate Computer Engineering students at the University of Ilorin (who are Raqib's peers). Subsequently, the model underwent fine-tuning using BERT and KerasNLP techniques, resulting in an impressive accuracy score of 96%. The objective is to uncover sentiments expressed in the feedback and gain a comprehensive understanding of student perceptions and satisfaction regarding their educational experience.  
To utilize the application, simply input your text, and it will promptly reveal the underlying sentiment.
The app also has Exploratory Data Analysis capabilities.
""")

# Add text input
text = st.text_input("Enter your text:")

# Add a clear button
if st.button("Clear Output"):
    text = ''

# Predict the sentiment using the loaded model
with st.spinner("Loading Output.."):
    if text:
        preprocessed_text = preprocessor([text])
        sentiment = model.predict(preprocessed_text)

        sentiment_categories = ["Negative", "Positive"]
        sentiment_label = sentiment_categories[np.argmax(sentiment)]
        confidence = (100 * np.max(sentiment)).round(2)

        if sentiment_label == "Positive":
            st.success(f"The sentiment of your text is: {sentiment_label} with a {confidence} percent confidence.")
        else:
            st.error(f"The sentiment of your text is: {sentiment_label} with a {confidence} percent confidence.")
            
            
df1['Sentiments'] = df1['Sentiments'].replace({
    0: 'negative',
    1:'positive'
})    
    
st.markdown(
    f"""
    <style>
        div.stButton > button:first-child {{
            background-color: #636EFA;
            color: white;
            font-weight: bold;
            font-size: 18px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# separate view for visualizations
if st.button("Explore Visualizations"):
    # Create a subpage for visualizations
    with st.expander("Sentiments Distribution"):
        sentiment_counts = df1['Sentiments'].value_counts()
        fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, labels={'x': 'Sentiment', 'y': 'Count'})
        fig.update_layout(
            xaxis=dict(type='category'),
            title="Distribution of Sentiments",
            xaxis_title="Sentiment",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)

    with st.expander("Sentiments Distribution (Pie Chart)"):
        label_data = df1['Sentiments'].value_counts()
        fig = px.pie(label_data, values=label_data.values, names=label_data.index, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Sentiments Distribution (Pie Chart)")
        st.plotly_chart(fig)

    with st.expander("Word Cloud Visualization"):
        all_feedback = ' '.join(X_preprocessed)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_feedback)
        fig = px.imshow(wordcloud)
        fig.update_layout(title='Word Cloud of Overall Feedback')
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig)
        
    with st.expander("Course Difficulty"):
        course_difficulty_counts = df1['Course Difficulty'].value_counts()
        fig = px.bar(course_difficulty_counts, x=course_difficulty_counts.index, y=course_difficulty_counts.values, labels={'x': 'Course Difficulty', 'y': 'Count'})
        fig.update_layout(
            xaxis=dict(type='category'),
            title="Feedback Count by Course Difficulty",
            xaxis_title="Course Difficulty",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)

    with st.expander("Feedback Count by Course Code"):
        course_code_counts = df1['Course Code'].value_counts()
        fig = px.bar(course_code_counts, x=course_code_counts.index, y=course_code_counts.values, labels={'x': 'Course Code', 'y': 'Count'})
        fig.update_layout(
            xaxis=dict(type='category'),
            title="Feedback Count by Course Code",
            xaxis_title="Course Code",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)

    with st.expander("Gender Distribution"):
        gender_counts = df1['Gender'].value_counts()
        fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Gender Distribution")
        st.plotly_chart(fig)
        
    with st.expander("Most Frequently Used Words"):
        from collections import Counter
        word_frequency = Counter(" ".join(df1['Feedback']).split()).most_common(30)
        word_df = pd.DataFrame(word_frequency, columns=['Word', 'Frequency'])
        fig = px.bar(word_df, x='Frequency', y='Word', orientation='h', title='Top 30 Most Frequently Used Words')
        st.plotly_chart(fig)
        
    with st.expander("Course Code distribution by Sentiment distribution"):
        fig = px.histogram(df1, x='Course Code', color='Sentiments', title='Course Code distribution by Sentiment distribution')
        fig.update_xaxes(title='Course Code')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)
        
    with st.expander("Sentiment Distribution by Course Difficulty"):
        fig = px.histogram(df1, x='Course Difficulty', color='Sentiments', 
                           title='Sentiment Distribution by Course Difficulty',
                           category_orders={"Course Difficulty": ['Easy', 'Moderate', 'Challenging', 'Difficult']})
        fig.update_xaxes(title='Course Difficulty')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)
        
    with st.expander("Sentiment Distribution by Gender"):
        fig = px.histogram(df1, x='Gender', color='Sentiments', title='Sentiment Distribution by Gender')
        fig.update_xaxes(title='Gender')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)
        
    with st.expander("Distribution of Word Count for different levels of Course Difficulty"):
        fig = px.box(df1, x='Course Difficulty', y='Word_Count', 
                     title='Distribution of Word Count for different levels of Course Difficulty',
                     category_orders={"Course Difficulty": ['Easy', 'Moderate', 'Challenging', 'Difficult']})
        fig.update_xaxes(title='Course Difficulty')
        fig.update_yaxes(title='Word Count')
        st.plotly_chart(fig)
        
    with st.expander("Distribution of Study Hours (per week) and Overall Satisfaction"):
        fig = px.scatter(df1, x='Study Hours (per week)', y='Overall Satisfaction')
        fig.update_layout(
            title="Distribution of Study Hours (per week) and Overall Satisfaction",
            xaxis_title="Study Hours (per week)",
            yaxis_title="Overall Satisfaction",
        )
        st.plotly_chart(fig)
        
    with st.expander("Sentiment score vs. Overall Satisfaction"):
        fig = px.scatter(df1, x='Sentiment_Scores', y='Overall Satisfaction')
        fig.update_layout(
            title="Sentiment score vs. Overall Satisfaction",
            xaxis_title="Sentiment Score",
            yaxis_title="Overall Satisfaction",
        )
        st.plotly_chart(fig)
        
    with st.expander("Correlation between Features"):
        correlation_matrix = df1[['Study Hours (per week)', 'Overall Satisfaction', 'Hour', 'Sentiment_Scores']].corr()
        fig = px.imshow(correlation_matrix, labels=dict(x="Features", y="Features", color="Correlation"))
        fig.update_layout(
            title="Correlation between variables in the dataset",
            xaxis_title="Features",
            yaxis_title="Features",
        )
        st.plotly_chart(fig)
        
# Sentiment Summary
st.title("Sentiment Summary")

# summary statistics
average_sentiment = df1["Sentiment_Scores"].mean()
positive_feedback_count = (df1["Sentiments"] == "positive").sum()
negative_feedback_count = (df1["Sentiments"] == "negative").sum()

# Display summary statistics
st.write(f"Average Sentiment Score: {average_sentiment:.2f}")
st.write(f"Number of Positive Feedback: {positive_feedback_count}")
st.write(f"Number of Negative Feedback: {negative_feedback_count}")

# footer

# line separator
st.markdown('<hr style="border: 2px solid #ddd;">', unsafe_allow_html=True)

# footer text
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        Developed by <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a>
    </div>
    """,
    unsafe_allow_html=True
)
