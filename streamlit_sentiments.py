import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import string
import tensorflow as tf
from tensorflow import keras
import keras_nlp
from transformers import BertTokenizer, TFBertForSequenceClassification
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Configure Streamlit page
st.set_page_config(
    page_title="Raqib Sentiments App",
    page_icon="R.png",
    layout="wide",
    # background_color="#F5F5F5"
)

# Load the exported data using st.cache
@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv('exported_sentiments.csv')

df1 = load_data()

# Encode the target labels
# df1['Sentiments'] = df1['Sentiments'].replace({
#     'negative': 0,
#     'positive': 1
# })

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

# # Convert labels to one-hot encoded format
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

# Add the subheading and image in a container
with st.container():
    st.subheader("Sentiment Analysis App")
    st.image("sentim.jpg")

# Add an introductory paragraph
st.markdown("""
This web application is a sentiment analysis tool developed by Raqib (popularly known as raqibcodes). It can be used to determine whether user-entered text has a Positive or Negative sentiment. The underlying text classification model was trained on feedback data collected from 300 level undergraduate Computer Engineering students at the University of Ilorin (who are Raqib's peers). Subsequently, the model underwent fine-tuning using BERT and KerasNLP techniques, resulting in an impressive accuracy score of 96%. The objective is to uncover sentiments expressed in the feedback and gain a comprehensive understanding of student perceptions and satisfaction regarding their educational experience.  
To utilize the application, simply input your text, and it will promptly reveal the underlying sentiment.
The app also has Exploratory Data Analysis capabilities.
""")

# Create containers for sideboxes
course_code_container = st.empty()
previous_exp_container = st.empty()
gender_container = st.empty()
attendance_container = st.empty()
difficulty_container = st.empty()
study_hours_container = st.empty()
satisfaction_container = st.empty()
department_container = st.empty()

# Initialize variables to store sidebox values
course_code = None
previous_exp = None
gender = None
attendance = None
difficulty = None
study_hours = None
satisfaction = None
department = None

# Get values from sideboxes
course_code = course_code_container.selectbox("Course Code", ['Select Course Code', 'CPE 321', 'CPE 311', 'CPE 341', 'CPE 381', 'CPE 331', 'MEE 361', 'GSE 301'])
previous_exp = previous_exp_container.selectbox("Previous Experience", ['Select Option', "Yes", "No"])
gender = gender_container.selectbox("Gender", ['Select Gender', 'Male', 'Female'])
attendance = attendance_container.selectbox("Attendance", ['Select Attendance', 'Regular', 'Irregular', 'Occasional'])
difficulty = difficulty_container.selectbox("Course Difficulty", ['Select Difficulty', 'Easy', 'Difficult', 'Challenging', 'Moderate'])
study_hours = study_hours_container.slider("Study Hours (per week)", 0, 24)
satisfaction = satisfaction_container.slider("Overall Satisfaction", 1, 10)
department = department_container.selectbox("Department", ['Select Option', "Yes", "No"])

# Add text input
text = st.text_input("Enter your text:")

if st.button("Submit Predictions"):
    # Check if all required fields are filled
    if not text or course_code == 'Select Course Code' or previous_exp == 'Select Option' or gender == 'Select Gender' or \
            attendance == 'Select Attendance' or difficulty == 'Select Difficulty' or study_hours is None or \
            satisfaction is None or department == 'Select Option':
        st.warning("Please fill in all the required fields before submitting predictions.")
    else:
        # Predict the sentiment with a spinner
        with st.spinner("Loading Output.."):
            preprocessed_text = preprocessor([text])
            sentiment = classifier.predict(preprocessed_text)

            sentiment_categories = ["negative", "positive"]
            sentiment_label = sentiment_categories[np.argmax(sentiment)]
            confidence = (100 * np.max(sentiment)).round(2)

            if sentiment_label == "positive":
                st.success(f"The sentiment of your text is: {sentiment_label} with a {confidence} percent confidence.")
            else:
                st.error(f"The sentiment of your text is: {sentiment_label} with a {confidence} percent confidence.")

            # Append the new row to the DataFrame with numerical label
            new_row = {
                'Course Code': course_code,
                'Feedback': text,
                'Previous Experience': previous_exp,
                'Gender': gender,
                'Attendance': attendance,
                'Course Difficulty': difficulty,
                'Study Hours (per week)': study_hours,
                'Overall Satisfaction': satisfaction,
                'Department': department,
                'Date': datetime.today().strftime('%Y-%m-%d'),
                'Time': datetime.now().strftime('%H:%M:%S'),
                'Hour': None,  # Hour will be extracted from Time
                'Processed_Feedback': None, 
                'Char_Count': None,
                'Word_Count': None,
                 'Sentiments': 1 if sentiment_label == "positive" else 0
            }

            # Extract Hour from Time
            new_row['Hour'] = pd.to_datetime(new_row['Time']).hour if new_row['Time'] else None

            # Process the text
            processed_text = preprocess_text(text)
            new_row['Processed_Feedback'] = processed_text
            new_row['Char_Count'] = len(processed_text)
            new_row['Word_Count'] = len(processed_text.split())

            df1 = pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)

            # Save the updated dataset to the CSV file
            try:
                df1.to_csv('exported_sentiments.csv', index=False)
                st.success("Data saved successfully.")

                # Save to CSV in Google Drive
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                credentials = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
                gc = gspread.authorize(credentials)
                spreadsheet = gc.open("Your Spreadsheet Title")
                worksheet = spreadsheet.sheet1
                worksheet.clear()  # Clear the existing data
                worksheet.append_table(list(df1.columns))
                worksheet.append_table(df1.values.tolist())
                st.success("Data saved to Google Drive successfully.")
            except Exception as e:
                st.error(f"Error saving data: {str(e)}")
                
             # Update Sentiments values in real-time
            # df1['Sentiments'] = df1['Sentiments'].replace({0: 'negative', 1: 'positive'})
    
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

 # Update summary statistics in real-time
average_sentiment = df1["Sentiment_Scores"].mean()
positive_feedback_count = (df1["Sentiments"] == 1).sum()
negative_feedback_count = (df1["Sentiments"] == 0).sum()

# Display updated summary statistics
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
