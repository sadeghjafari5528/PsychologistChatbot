import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_chatbot.settings')
django.setup()


import csv
from chatbot.models import Chat  # Replace 'your_app_name' with the actual name of your app

# Fetch all Chat records, sorted by username and creation time
chats = Chat.objects.select_related('user').exclude(message='').order_by('user__username', 'created_at')

# Define the CSV file path
csv_file_path = 'data/chats.csv'

# Define CSV header
header = [
    'username',
    'created_at',
    'message', 
    'response', 
    'emotion', 
    'disorder', 
    'validation', 
]

# Open the CSV file for writing
with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow(header)

    # Write the chat data
    for chat in chats:
        writer.writerow([
            chat.user.username,
            chat.created_at,
            chat.message,
            chat.response,
            chat.emotion,
            chat.disorder,
            chat.validation
        ])

print(f"Chat data has been saved to {csv_file_path}.")
