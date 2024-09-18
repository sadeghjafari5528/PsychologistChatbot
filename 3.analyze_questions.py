import pandas as pd

# Example dataframe
df = pd.read_csv('data/questions.csv')

# Define the list of unique questions
questions = list(df['question'].unique())

# Group by 'username' and 'question' and aggregate the answers
nested_answers = df.groupby(['username', 'question'])['answer'].apply(list).reset_index()

# Create a nested list for each user, with 9 items (one per question)
def create_nested_list(user_data):
    result = []
    for q in questions:
        answers = user_data[user_data['question'] == q]['answer']
        result.append(answers.iloc[0] if not answers.empty else [])
    return result

# Apply this for each user
final_result = nested_answers.groupby('username').apply(create_nested_list).reset_index(name='nested_answers')

# Display the result
print(final_result)
