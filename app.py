from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Example dataset of courses
courses_data = [
    {"course_name": "Intro to Python", "course_description": "Learn the basics of Python programming, including data types, control flow, and more."},
    {"course_name": "Advanced Python", "course_description": "In-depth look at Python, focusing on advanced concepts like decorators, generators, and multi-threading."},
    {"course_name": "Machine Learning 101", "course_description": "An introduction to machine learning techniques including supervised and unsupervised learning."},
    {"course_name": "Deep Learning with TensorFlow", "course_description": "Learn to implement deep learning algorithms using TensorFlow and Keras."},
    {"course_name": "Data Structures & Algorithms", "course_description": "Master fundamental data structures and algorithms for efficient coding interviews."},
    {"course_name": "Natural Language Processing", "course_description": "Explore NLP techniques such as tokenization, stemming, and sentiment analysis."},
    {"course_name": "Artificial Intelligence Basics", "course_description": "An introductory course on artificial intelligence and its real-world applications."},
    {"course_name": "Computer Vision with OpenCV", "course_description": "Learn computer vision techniques using OpenCV for image processing and object detection."},
    {"course_name": "SQL for Data Science", "course_description": "Learn SQL queries and database management for data science applications."},
    {"course_name": "Data Analysis with Pandas", "course_description": "Data analysis using the Pandas library in Python to manipulate and analyze data."},
    {"course_name": "Introduction to R", "course_description": "Learn the basics of R programming, data structures, and visualizations."},
    {"course_name": "Introduction to Machine Learning with Scikit-Learn", "course_description": "Learn machine learning algorithms and their implementation using Scikit-Learn."},
    {"course_name": "Big Data with Hadoop", "course_description": "Explore big data concepts and tools, focusing on Hadoop and MapReduce."},
    {"course_name": "Reinforcement Learning", "course_description": "An introduction to reinforcement learning and its applications."},
    {"course_name": "Deep Learning Specialization", "course_description": "Advanced course covering neural networks, CNNs, RNNs, and reinforcement learning."},
    {"course_name": "AWS Cloud Practitioner", "course_description": "Introduction to Amazon Web Services (AWS) for cloud computing and storage."},
    {"course_name": "Java Programming Basics", "course_description": "Learn the fundamentals of Java programming, including classes, objects, and basic syntax."},
    {"course_name": "JavaScript for Beginners", "course_description": "A beginner’s guide to JavaScript, covering syntax, functions, and DOM manipulation."},
    {"course_name": "Web Development with Django", "course_description": "Learn to build web applications using Django, focusing on models, views, and templates."},
    {"course_name": "Data Visualization with Matplotlib", "course_description": "Learn data visualization techniques using Python’s Matplotlib library."},
    {"course_name": "Blockchain Basics", "course_description": "Introduction to blockchain technology and its applications in finance and beyond."},
    {"course_name": "Cloud Computing with Azure", "course_description": "Learn about cloud computing principles and services offered by Microsoft Azure."},
    {"course_name": "Digital Marketing Fundamentals", "course_description": "Introduction to digital marketing strategies, social media, and SEO."},
    {"course_name": "Statistics for Data Science", "course_description": "Master statistical techniques and tools for data analysis."},
    {"course_name": "Cybersecurity Essentials", "course_description": "Learn the basics of cybersecurity, including network security and cryptography."},
    {"course_name": "Web Scraping with Python", "course_description": "Learn to extract data from websites using Python libraries like BeautifulSoup."},
    {"course_name": "Artificial Neural Networks", "course_description": "Study the theory and application of neural networks in machine learning."},
    {"course_name": "Data Science and Machine Learning Bootcamp", "course_description": "A bootcamp covering Python, data analysis, machine learning, and more."},
    {"course_name": "Game Development with Unity", "course_description": "Learn to build 2D and 3D games using the Unity engine."},
    {"course_name": "Ethical Hacking and Penetration Testing", "course_description": "Learn penetration testing and ethical hacking techniques for network security."},
    {"course_name": "Quantum Computing Basics", "course_description": "Introduction to quantum computing and its theoretical underpinnings."}
]

# Convert data to DataFrame
df_courses = pd.DataFrame(courses_data)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_courses["course_description"])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Course Recommendation Function
def recommend_courses(user_skills, num_recommendations=5):
    if not user_skills.strip():  # Handle empty input
        return ["Please enter at least one skill."]

    # Vectorize the user input
    user_skills_vec = tfidf_vectorizer.transform([user_skills])
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(user_skills_vec, tfidf_matrix).flatten()
    
    # Debugging: Print similarity scores
    print("\nUser Input:", user_skills)
    print("Similarity Scores:", similarity_scores)

    # Get indices of top recommended courses
    similar_courses_idx = similarity_scores.argsort()[-num_recommendations:][::-1]

    # Filter out zero-similarity results (irrelevant recommendations)
    recommended_courses = [df_courses.iloc[i]["course_name"] for i in similar_courses_idx if similarity_scores[i] > 0]

    if not recommended_courses:
        return ["No matching courses found. Try different skills!"]
    
    return recommended_courses

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        user_skills = request.form['skills']
        print("Received Skills:", user_skills)  # Debugging print
        recommendations = recommend_courses(user_skills, num_recommendations=5)
        print("Recommended Courses:", recommendations)  # Debugging print
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
