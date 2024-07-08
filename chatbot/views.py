from datetime import datetime
import numpy as np

from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.utils import timezone

import openai

from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat



from chatbot.disorder_detector.stress_detector import check_for_stress_in_text, load_stress_detector_model_tokenizer
from chatbot.emotion.emotion_detection import load_emotion_detector_model_tokenizer, predict_emotion_label, predict_emotion_of_texts
from chatbot.message_validator.message_validator import load_validator_model_and_tokenizer, predict_validator_labels

from dotenv import load_dotenv
import os
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
print('openai_api_key', openai_api_key)
openai.api_key = openai_api_key

validator_model, validator_tokenizer = load_validator_model_and_tokenizer()
emotion_model, emotion_tokenizer = load_emotion_detector_model_tokenizer()
disorder_tokenizer, disorder_model = load_stress_detector_model_tokenizer()


# def calculate_weighted_average(chats: list[Chat], feature: str, decay_factor: float = 0.9):
#     current_time = timezone.now().timestamp()

#     label_weighted_scores = {}
#     label_total_weights = {}

#     for chat in chats:
#         chat_time = chat.created_at().timestamp()
#         time_diff = current_time - chat_time
#         weight = np.exp(-decay_factor * time_diff)  # Exponential decay weight
        
#         if chat.label not in label_weighted_scores:
#             label_weighted_scores[chat.label] = 0
#             label_total_weights[chat.label] = 0

#         label_weighted_scores[chat.label] += chat.score * weight
#         label_total_weights[chat.label] += weight

#     # Step 4: Calculate Weighted Average Scores
#     label_weighted_averages = {
#         label: label_weighted_scores[label] / label_total_weights[label]
#         for label in label_weighted_scores
#     }

#     # Step 5: Determine the Max Weighted Average Score
#     max_label = max(label_weighted_averages, key=label_weighted_averages.get)
#     max_weighted_average_score = label_weighted_averages[max_label]
# #-------------------------------------------------------------------------
#     current_time = timezone.now().timestamp()
#     weighted_scores = []
#     total_weight = 0

#     for chat in chats:
#         chat_time = chat.created_at().timestamp()
#         time_diff = current_time - chat_time
#         weight = np.exp(-decay_factor * time_diff)
#         weighted_scores.append(getattr(chat, feature) * weight)
#         total_weight += weight

#     weighted_average = sum(weighted_scores) / total_weight if total_weight != 0 else 0
#     return weighted_average



def ask_openai(chat_obj: Chat, chat_history, window_size: int = None):
    if window_size:
        chat_history = chat_history.order_by('-id')[:window_size]
    
    messages = list()
    for chat in chat_history:
        messages.append({"role": "user", "content": chat.message})
        messages.append({"role": "system", "content": chat.response})

    # average_emotion_prob = calculate_weighted_average(list(chat_history) + [chat_obj], 'emotion')
    # average_disorder_prob = calculate_weighted_average(list(chat_history) + [chat_obj], 'disorder')
    prompt = """
Previous messages are the chat history between a paitient and a pychologist, 
"""
    messages.append({"role": "user", "content": chat_obj.message})

    response = openai.ChatCompletion.create(
        model = "gpt-4-turbo",
        # prompt = message,
        # max_tokens=150,
        # n=1,
        # stop=None,
        # temperature=0.7,
        messages=messages
    )
    answer = response.choices[0].message.content.strip()
    return answer

# Create your views here.

def chatbot(request):
    chats = Chat.objects.filter(user=request.user)


    if request.method == 'POST':
        message = request.POST.get('message')
        disorder = check_for_stress_in_text(message, disorder_model, disorder_tokenizer)
        emotion = predict_emotion_label(message, emotion_model, emotion_tokenizer)
        chat = Chat(
            user=request.user,
            message=message,
            # response=response,
            created_at=timezone.now,
            emotion=emotion,
            disorder=disorder,
            # validation_label='',
        )
        for i in range(5):
            response = ask_openai(chat, chat_history=chats, window_size=20)
            validation = predict_validator_labels(
                response,
                validator_model,
                validator_tokenizer
            )
            if len(validation) == 0:
                break

        chat.validation = validation
        chat.response = response
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})


def login(request):
    if request.method=='POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1==password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('chatbot')
            except:
                error_message = 'Error creating account'
            return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = "Password don't match" 
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('login')