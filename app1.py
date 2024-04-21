import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to load and preprocess the data
def load_data(file):
    df = pd.read_csv(file,delimiter=";")
    return df

# Function to process the input and get the most similar question
def get_most_similar_question(new_sentence, questions, answers, vectorizer, tfidf_matrix):
    new_tfidf = vectorizer.transform([new_sentence])

    similarities = cosine_similarity(new_tfidf, tfidf_matrix)

    most_similar_index = np.argmax(similarities)

    similarity_percentage = similarities[0, most_similar_index] * 100

    return answers[most_similar_index], similarity_percentage

# Function to generate response
def AnswertheQuestion(new_sentence, questions, answers, vectorizer, tfidf_matrix):
    most_similar_answer, similarity_percentage = get_most_similar_question(new_sentence, questions, answers, vectorizer, tfidf_matrix)
    if similarity_percentage > 70:
        response = {
            'answer': most_similar_answer
        }
    else:
        response = {
            'answer': 'Sorry, I am not aware of this information :('
        }

    return response

# Streamlit app
def main():
    st.title("Q&A Chatbot")
    st.write("Upload a CSV file with questions and answers.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(questions)

        # Ask question
        user_question = st.text_input("Ask your question here:")
        if st.button("Ask"):
            if user_question:
                response = AnswertheQuestion(user_question, questions, answers, vectorizer, tfidf_matrix)
                st.write("Answer:", response['answer'])

if __name__ == "__main__":
    main()
