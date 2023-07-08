# Sentiment Analysis on Student Feedback

## Description
This project focuses on sentiment analysis of student feedback in engineering education. It is particularly directed at my engineering class in the university. It aims to gain insights from feedback data, improve education quality, and enhance the student experience. The project employs natural language processing techniques, topic modeling, and machine learning algorithms to analyze sentiments expressed by the students. It provides actionable recommendations based on sentiment analysis results.

## Key Features
- Introduction: Overview of the project.
- Data Collection: Collect student feedback data using [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSeInhWFxECdegDDIYo7uY3-U-JHYyUDkTBQBw-KJxIvzYg_yA/viewform).
- Data Cleaning: Clean and preprocess the collected data for analysis.
- Data Preprocessing: Perform text preprocessing techniques such as tokenization, stop word removal, and stemming/lemmatization.
- Aspect-Based Sentiment Analysis: Analyze the sentiment of specific aspects or keywords related to engineering education.
- Topic Modeling: Identify key themes and topics from the feedback data.
- Emotion Detection: Detect and analyze emotions expressed in the feedback.
- Visualizations: Generate visualizations to understand sentiment distribution and trends in the feedback.
- Conclusions

## Installation
1. Clone the repository: `git clone https://github.com/Abdulraqib20/Sentiment-Analysis-with-Student-Feedback.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
1. Collect and clean the feedback data.
2. Preprocess the data using text preprocessing techniques.
3. Perform aspect-based sentiment analysis on specific topics or keywords.
4. Apply topic modeling to identify key themes and topics in the feedback.
5. Detect and analyze emotions expressed in the feedback.
6. Generate visualizations to understand sentiment distribution and feedback trends.
7. Interact with the Power BI [Dashboard](https://app.powerbi.com/links/AupqjRXtlh?ctid=66b3f0c2-8bc6-451e-9603-986f618ae682&pbi_source=linkShare&bookmarkGuid=486f8ecd-e9fa-4f5e-9bfd-c684a5528ddb).
8. Read the article I wrote on the project [here](https://medium.com/@abdulraqibshakir03/sentiment-analysis-on-student-feedback-in-engineering-education-55a913dd7967?source=user_profile---------2----------------------------).

## Results and Findings
The sentiment analysis of student feedback in engineering education has yielded valuable insights and recommendations for improvement. The sentiment distribution indicates a majority of `Neutral` feedback (suggesting a balanced perspective or lack of strong sentiment towards their educational experience) followed by `Positive` and `Negative` sentiments. The prevalence of Neutral sentiments in the student feedback sentiment analysis may also indicate that students are providing objective observations or factual statements without expressing a clear positive or negative sentiment. Furthermore, it was observed that a majority of students had no previous experience, adding to the context of the analysis.

Gender disparity shows both male and female students expressing negative sentiments, with females expressing more positive sentiments and males having more neutral sentiments which highlight the importance of considering gender as a factor in understanding and addressing the sentiment dynamics in student feedback. 

Variation across courses highlights specific strengths or areas for improvement, with `CPE 321` being the most challenging. `CPE 341` and `CPE 311` received lower sentiment scores, while `CPE 311` had the highest sentiment score. It was also noted that the easy courses had the most number of positive sentiments. Correlations reveal the alignment of sentiment score with overall satisfaction and perfect correlation with emotion polarity. 

Additionally, the high correlation between study hours and overall satisfaction implies that the amount of time students dedicate to studying may positively influence their overall satisfaction with the courses they take. 

Topic modeling uncovers key themes discussed by students. These findings can serve as a basis for informed decision-making, allowing the department to address necessary concerns, capitalize on strengths, and continuously enhance the quality of education provided to students. The sentiment analysis serves as a foundation for continuous improvement in engineering education with targeted interventions required for courses with more negative sentiments, particularly `CPE 321` ensuring a fulfilling and satisfactory learning conditions for the students.

Lastly, I built a model which achieved impressive accuracy rates of 68% in cross-validation and 75% in testing. This project has deepened my understanding of ML and fueled my passion for using technology to understand and predict human emotions.

## Limitations and Future Work
One of the limitations of this project is the relatively small size of the dataset. The data collected for sentiment analysis on student feedback in engineering education may not represent the entire student population or provide a comprehensive view of sentiments. This limitation could affect the generalizability of the findings and the accuracy of the sentiment analysis results.

To address this limitation, future work could involve collecting a larger and more diverse dataset to improve the robustness and reliability of the sentiment analysis. Additionally, exploring external data sources or incorporating data from multiple educational institutions could provide a broader perspective on student sentiments and enhance the overall analysis.

## Contributing
Contributions are welcome! If you would like to contribute to the project, feel free to reach out to me. Together, we can enhance the analysis and make a positive impact on engineering education.

## Contact
For any questions or inquiries, please contact abdulraqibshakir03@gmail.com.
