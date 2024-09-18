import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_chatbot.settings')
django.setup()


import csv
from chatbot.models import Questionnaire

# Fetch all Questions records, sorted by username and creation time
questions = Questionnaire.objects.select_related('user').filter(is_phq_9=True).order_by('user__username', 'created_at')

# Define the CSV file path
csv_file_path = 'data/questions.csv'

# Define CSV header
header = [
    'username',
    'created_at',
    'question', 
    'answer',
]

# Open the CSV file for writing
with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow(header)

    # Write the chat data
    for question in questions:
        writer.writerow([
            question.user.id,
            question.created_at,
            question.question,
            question.answer,
        ])

print(f"Question data has been saved to {csv_file_path}.")
