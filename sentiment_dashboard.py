import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Initialize the Dash app
app = dash.Dash(__name__)

df = pd.read_csv('Sentiment Analysis on Student Feedback.csv')

# Define the layout of the dashboard
app.layout = html.Div(
    children=[
        html.H1("Sentiment Analysis Dashboard"),
        
        # Section 1: Course Code Selection
        html.Div([
            html.Label("Select a Course Code:"),
            dcc.Dropdown(
                id="course-code-dropdown",
                options=[
                    {"label": course_code, "value": course_code}
                    for course_code in df['Course Code'].unique()
                ],
                value=df['Course Code'].unique()[0]
            )
        ]),
        
        # Section 2: Word Cloud
        html.Div([
            html.H2("Word Cloud"),
            html.Div(id="word-cloud")
        ])
    ]
)

# Define the callback for updating the word cloud
@app.callback(
    Output("word-cloud", "children"),
    Input("course-code-dropdown", "value")
)
def update_word_cloud(course_code):
    filtered_data = df[df['Course Code'] == course_code]
    feedback_text = ' '.join(filtered_data['Processed_Feedback'])
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400).generate(feedback_text)
    
    # Display the word cloud using matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("word_cloud.png")  # Save the word cloud as an image
    plt.close()
    
    return html.Img(src="word_cloud.png", style={'width': '100%'})

# Run the application
if __name__ == "__main__":
    app.run_server(debug=True)
