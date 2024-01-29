
import pandas as pd
import random


def get_extended_question():
    df = pd.read_excel(
        'follow_up_questions\datasets\PHQ-9_Extended_Sample_Questions.xlsx')
    all_questions = {}
    key = ""
    # Iterate through rows
    for index, row in df.iterrows():
        question = row['Original Question in PHQ9']
        related_question = row['Related Questions']
        if (question not in all_questions):
            all_questions[question] = []
            key = question
        all_questions[key].append(related_question)

    return all_questions


def ask_question(text):
    questions = get_extended_question()

    random_question = list(questions.keys())[random.randint(0, 8)]
    related_question = random.randint(0, len(questions[random_question])-1)
    return questions[random_question][related_question]




print(ask_question("hello"))
