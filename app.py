import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv('qna.csv',encoding = 'utf-8',delimiter=';')
print(df)
questions = df['question'].tolist()
print(questions)
answers = df['answer'].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def get_most_similar_question(new_sentence):
    new_tfidf = vectorizer.transform([new_sentence])

    similarities = cosine_similarity(new_tfidf,tfidf_matrix)

    most_similar_index = np.argmax(similarities)

    similarity_percentage = similarities[0, most_similar_index]*100

    return answers[most_similar_index], similarity_percentage

def AnswertheQuestion(new_sentence):
    most_similar_answer, similarity_percentage = get_most_similar_question(new_sentence)
    if similarity_percentage > 70:
        response = {
            'answer': most_similar_answer
        
        }
    else:
        response = {
            'answer': 'Sorry, I am not aware of this information :('
        }

    return response

print(AnswertheQuestion('Who is the Ninad'))
    