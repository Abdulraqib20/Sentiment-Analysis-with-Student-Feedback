import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import tensorflow as tf
from tensorflow import keras
import keras_nlp
from transformers import BertTokenizer, TFBertForSequenceClassification
import nltk
nltk.download('stopwords')
nltk.download('punkt')
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

# Create a Streamlit app
st.set_page_config(
    page_title="Raqib Sentiments App",
    page_icon="R.png",
    layout="wide",
    # background_color="#F5F5F5"
)

# Add the subheading and image in a container
with st.container():
    st.subheader("Raqib's Sentiment Analysis WebApp")
    st.image("sentim.jpg")

# Add an introductory paragraph
st.markdown("""
This web application is a sentiment analysis tool developed by Abdulraqib (raqibcodes). It can be used to determine whether user-entered text has a positive or negative sentiment. The underlying text classification model was trained on feedback data collected from 300 level undergraduate Computer Engineering students at the University of Ilorin (my classmates). Subsequently, the model underwent fine-tuning using BERT and KerasNLP techniques, resulting in an impressive accuracy score of 96%. The objective is to uncover sentiments expressed in the feedback and gain a comprehensive understanding of student perceptions and satisfaction regarding their educational experience.  
To utilize the application, simply input your text, and it will promptly reveal the underlying sentiment.
""")

# Add text input
text = st.text_input("Enter your text:")

# Add a clear button
if st.button("Clear"):
    text = ''

# Predict the sentiment with a spinner
with st.spinner("Loading Output.."):
    if text:
        preprocessed_text = preprocessor([text])
        sentiment = classifier.predict(preprocessed_text)

        sentiment_categories = ["Negative", "Positive"]
        sentiment_label = sentiment_categories[np.argmax(sentiment)]
        confidence = (100 * np.max(sentiment)).round(2)

        if sentiment_label == "Positive":
            st.success(f"The sentiment of your text is: {sentiment_label} with a {confidence} percent confidence.")
        else:
            st.error(f"The sentiment of your text is: {sentiment_label} with a {confidence} percent confidence.")
