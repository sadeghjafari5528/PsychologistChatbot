import datetime
import sys

import numpy as np

# import logging

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.utils import timezone, dateformat
import openai

from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat, Question, Questionnaire

# Import various models and utilities for the chatbot's functionality
from chatbot.disorder_detector.stress_detector import check_for_stress_in_text, load_stress_detector_model_tokenizer
from chatbot.emotion.emotion_detection import load_emotion_detector_model_tokenizer, predict_emotion_label, \
    predict_emotion_of_texts, label_dict
from chatbot.message_validator.message_validator import load_validator_model_and_tokenizer, predict_validator_labels

from dotenv import load_dotenv
import os

load_dotenv()

# Cooperation text that will be used in the chatbot response
cooperation_text = """
ما یه تیم هستیم متشکل از دوتا روانشناس و 5 متخصص هوش مصنوعی 
این چت بات رو طراحی کردیم با هدف …
و برامون خیلی مهمه که این چت بات بتونه اثر بخش باشه، برای همین از شما کمک میخوایم، 

چه کمکی؟ 

کاری که باید بکنید اینه که در مجموع ۳ نوبت پرسشنامه پر بکنید 
و روزی ۱۰ الی ۱۵ دقیقه هم از چت بات استفاده کنید و سعی کنید در زمینه افسردگی ازش کمک بگیرید.
و کل این فرایند فقط ۲ هفته طول میکشه. 

واما شما چی به دست میارین؟

میتونین به صورت رایگان از جلسات درمانی که براتون گذاشته میشه استفاده بکنید.


پس اگر به مدت ۲ هفته میتونین روزی ۱۰ الی ۱۵ دقیقه زمان بزارین، 
برای اطلاعات بیشتر:
به من در تلگرام @sadeghjafari1379 پیام بدید یا در لینکدین با من در ارتباط باشید.

💢فقط یه نکته ۴ روز بیشتر فرصت ندارین تا تصمیمتون رو به ما اعلام کنید.
"""


# Load OpenAI API key and models for emotion, stress, and validation detection
openai.api_key = os.getenv('OPENAI_API_KEY')
validator_model, validator_tokenizer = load_validator_model_and_tokenizer()
emotion_model, emotion_tokenizer = load_emotion_detector_model_tokenizer()
disorder_tokenizer, disorder_model = load_stress_detector_model_tokenizer()

# logger = logging.getLogger('django')


# Function to calculate the weighted average of emotion or disorder over the chat history
def calculate_weighted_average(chats: list[Chat], feature: str, decay_factor: float = 0.9):
    chats = [chat for chat in chats if chat.message != '']
    weighted_average = dict()

    # Iterate over each label in the first chat's feature (emotion/disorder)
    for label in getattr(chats[0], feature).keys():
        total_weight = 0
        weighted_scores = list()

        # Calculate weighted score for each chat based on the time elapsed
        for chat in chats:
            if chat.response == cooperation_text:
                continue
            day_diff = (timezone.now() - chat.created_at).days
            weight = np.exp(-decay_factor * day_diff)
            weighted_scores.append(getattr(chat, feature)[label] * weight)
            total_weight += weight

        # Calculate the weighted average
        weighted_average[label] = sum(weighted_scores) / total_weight if total_weight != 0 else 0
    return weighted_average


# Function to interact with OpenAI API to get a response based on chat history and patient message
def ask_openai(chat_obj: Chat, chat_history, window_size: int = None):
    chat_history_size = len(chat_history)
    if window_size:
        chat_history = chat_history.order_by('-id')[:window_size]

    messages = list()

    # Prepare the chat history to be sent to OpenAI
    for chat in chat_history:
        messages.append({"role": "user", "content": chat.message})
        messages.append({"role": "system", "content": chat.response})


    # If there are more than 3 chats, include emotional and disorder statuses in the prompt
    if chat_history_size > 3:
        average_emotion_prob = calculate_weighted_average(list(chat_history) + [chat_obj], 'emotion')
        average_disorder_prob = calculate_weighted_average(list(chat_history) + [chat_obj], 'disorder')

        prompt = f"""
The previous messages are the chat history between a patient and a psychologist.
Suppose you are a professional psychologist. Based on the following information,
respond to the patient with a short message.(Prevent to say 'Hi' in each message. And only speak in persian)

Emotional status: {average_emotion_prob}
Mental disorder status: {average_disorder_prob}
Patient message: {chat_obj.message}

Speak more sincerely and informaly and never tell the user what her/him stress and emotion level is like and don't speak about it,
only know it to answer properly.
"""
    else:

        # If chat history is small, only include the patient message
        prompt = f"""
The previous messages are the chat history between a patient and a psychologist.
Suppose you are a professional psychologist. Based on the following information,
respond to the patient with a short message.(Prevent to say 'Hi' in each message. And only speak in persian)

Patient message: {chat_obj.message}
"""

    messages.append({"role": "user", "content": prompt})

    # Request response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        # prompt = message,
        # max_tokens=150,
        # n=1,
        # stop=None,
        # temperature=0.7,
        messages=messages
    )
    answer = response.choices[0].message.content.strip()
    return answer


def gad_7_questions():
    return [
        "در طول دو هفته گذشته، چند بار احساس کردید که عصبی، نگران یا بی‌قرار هستید؟    \n عدد گزینه را واردکنید :               گزینه 0) هرگز       گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه 3) تقریبا هرروز",
        "در طول دو هفته گذشته، چند بار قادر نبودید،اضطراب یا نگرانی‌های خود را کنترل کنید ؟    \n عدد گزینه را واردکنید :               گزینه 0) هرگز       گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز ",
        "در طول دو هفته گذشته، چند بار به قدری نگران بوده‌اید که نتوانسته‌اید روی چیزهای دیگر تمرکز کنید  \n عدد گزینه را واردکنید :               گزینه 0) هرگز      گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز    ",
        "در طول دو هفته گذشته،  چند بار به قدری عصبی  بوده ‌اید که نتوانسته ‌اید آرام باشید ؟        \n  عدد گزینه را واردکنید :         گزینه 0) هرگز،       گزینه 1) چندروز،             گزینه 2) بیش از نصف روز ها            گزینه 3) تقریبا هرروز ",
        " در طول دو هفته گذشته، چند بار به قدری نگران بوده‌اید که احساس کرده‌اید کارهایتان به کندی پیش می‌روند؟  \n   عدد گزینه را واردکنید :     گزینه 0) هرگز،       گزینه 1) چندروز،             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز  ",
        "در طول دو هفته گذشته، چند بار به قدری نگران بوده‌اید که احساس کرده‌اید باید دائماً حرکت کنید؟      \n عدد گزینه را واردکنید :               گزینه 0) هرگز       گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز  ",
        "در طول دو هفته گذشته، چند بار به قدری عصبی بوده‌اید که خوابیدن برایتان مشکل شده است؟         \n عدد گزینه را واردکنید :               گزینه 0) هرگز      گزینه 1) چندروز            گزینه 2) بیش از نصف روز ه               گزینه  3) تقریبا هرروز   ",
    ]


def phq_9_questions():
    # PHQ-9 questions related to depression symptoms
    return [
          """
    در طول دو هفته گذشته، چند بار احساس کرده‌اید که علاقه یا لذتی به فعالیت‌هایی که معمولاً از آنها لذت می‌بردید، ندارید؟ 
    <br> عدد گزینه را وارد کنید : 
    <br> گزینه 0) هرگز 
    <br> گزینه 1) چندروز 
    <br> گزینه 2) بیش از نصف روزها 
    <br> گزینه 3) تقریباً هرروز
    """
        ,
        """
           در طول دو هفته گذشته، چند بار احساس کرده‌اید که افسرده، غمگین یا ناامید هستید
           <br> عدد گزینه را واردکنید :
           <br> گزینه 0) هرگز
           <br> گزینه 1) چندروز
           <br> گزینه 2) بیش از نصف روز ها
           <br> گزینه 3) تقریبا هرروز
           """
        ,
        """
        در طول دو هفته گذشته، چند بار با مشکل در خوابیدن، خواب زیاد یا بیدار شدن در نیمه‌شب مواجه بوده‌اید؟
        <br> عدد گزینه را واردکنید :
        <br> گزینه 0) هرگز
        <br> گزینه 1) چندروز
        <br> گزینه 2) بیش از نصف روز ها
        <br> گزینه 3) تقریبا هرروز
        """
        ,
        """
        در طول دو هفته گذشته، چند بار احساس خستگی یا کمبود انرژی داشته‌اید؟
        <br> عدد گزینه را واردکنید :
        <br> گزینه 0) هرگز
        <br> گزینه 1) چندروز
        <br> گزینه 2) بیش از نصف روز ها
        <br> گزینه 3) تقریبا هرروز
        """,
        """
        در طول دو هفته گذشته، چند بار اشتهای شما کاهش یا افزایش یافته است؟
        <br> عدد گزینه را واردکنید :
        <br> گزینه 0) هرگز
        <br> گزینه 1) چندروز
        <br> گزینه 2) بیش از نصف روز ها
        <br> گزینه 3) تقریبا هرروز
        """
        ,
        """
        در طول دو هفته گذشته، چند بار احساس کرده‌اید که خود را یک شکست‌خورده می‌دانید یا احساس کرده‌اید که خود یا خانواده‌تان را ناامید کرده‌اید؟
        <br> عدد گزینه را واردکنید :
        <br> گزینه 0) هرگز
        <br> گزینه 1) چندروز
        <br> گزینه 2) بیش از نصف روز ها
        <br> گزینه 3) تقریبا هرروز
        """
        ,
        """
        در طول دو هفته گذشته، چند بار با مشکل تمرکز کردن بر روی چیزهایی مثل خواندن روزنامه یا تماشای تلویزیون مواجه بوده‌اید؟
        <br> عدد گزینه را واردکنید :
        <br> گزینه 0) هرگز
        <br> گزینه 1) چندروز
        <br> گزینه 2) بیش از نصف روزها
        <br> گزینه 3) تقریبا هرروز
        """,
        """
        در طول دو هفته گذشته، چند بار حرکت یا صحبت شما به قدری کند بوده است که دیگران متوجه آن شده‌اند؟ یا برعکس، چند بار به قدری بی‌قرار بوده‌اید که نمی‌توانستید آرام بنشینید؟
        <br> عدد گزینه را واردکنید :
        <br> گزینه 0) هرگز
        <br> گزینه 1) چندروز
        <br> گزینه 2) بیش از نصف روزها
        <br> گزینه 3) تقریبا هرروز
        """,
        """
        در طول دو هفته گذشته، چند بار احساس کرده‌اید که بهتر است خودتان را آسیب بزنید یا خودکشی کنید؟
        <br> عدد گزینه را واردکنید :
        <br> گزینه 0) هرگز
        <br> گزینه 1) چندروز
        <br> گزینه 2) بیش از نصف روزها
        <br> گزینه 3) تقریبا هرروز
        """
    ]


def chatbot(request):
    user = request.user
    if not user.is_authenticated:
        return redirect('login')

    # Send cooperation text if the user hasn't received it yet
    if not Chat.objects.filter(response=cooperation_text, user=user).exists():
        chat = Chat(
            user=user,
            message='',
            response=cooperation_text,
            created_at=timezone.now(),
            emotion={v:0 for v in label_dict.values()},
            disorder={"Not Stressed": 0, "Stressed": 0},
            validation=[],
            gad_7=False,
            phq_9=False,
        )
        chat.save()
    today = timezone.now().date()

    # Fetch the last question record, or create one if it doesn't exist
    question_record = Question.objects.filter(user=user).order_by('-created_at').first()
    if not question_record:
        question_record = Question.objects.create(user=user)

    # Check if it's time for a new questionnaire (weekly)
    if (timezone.now() - question_record.created_at).days >= 7:
        question_record, created = Question.objects.get_or_create(user=user, created_at__date=today)

    if request.method == 'POST':
        message = request.POST.get('message')

        # If the user is staff, handle the PHQ-9 questionnaire
        if user.is_staff:
            if question_record.phq_9_count == 0 and not question_record.phq_9_completed:
                first_question = phq_9_questions()[0]
                Questionnaire.objects.create(
                    user=user,
                    created_at=timezone.now(),
                    question=first_question,
                    answer='',
                    phq_9_number=1,
                    is_gad_7=False,
                    is_phq_9=True
                )
                question_record.phq_9_count += 1
                question_record.save()
                return JsonResponse({'response': first_question})

            if (question_record.phq_9_count > 0 and not question_record.phq_9_completed) :
                try:
                    last_questionnaire_entry = Questionnaire.objects.filter(
                        user=user,
                        phq_9_number=question_record.phq_9_count,
                        is_gad_7=False,
                        is_phq_9=True
                    ).order_by('created_at').last()


                    if last_questionnaire_entry.answer == '':
                        last_questionnaire_entry.answer = message
                        last_questionnaire_entry.save()
                except Questionnaire.DoesNotExist:
                    pass

            if not question_record.phq_9_completed:
                questions = phq_9_questions()
                if question_record.phq_9_count <= len(questions):
                    if question_record.phq_9_count < len(questions):
                        current_question = questions[question_record.phq_9_count]

                        Questionnaire.objects.create(
                            user=user,
                            created_at=timezone.now(),
                            question=current_question,
                            answer='',
                            phq_9_number=question_record.phq_9_count + 1,
                            is_gad_7=False,
                            is_phq_9=True
                        )
                    if question_record.phq_9_count < 9:
                        question_record.phq_9_count += 1
                        question_record.save()

                    last_questionnaire_entry = Questionnaire.objects.filter(
                        user=user,
                        phq_9_number=question_record.phq_9_count,
                        is_gad_7=False,
                        is_phq_9=True
                    ).order_by('created_at').last()

                    if question_record.phq_9_count == len(questions) and not (last_questionnaire_entry.answer == ''):

                        question_record.phq_9_completed = True
                        question_record.save()
                        return JsonResponse({'message': message,
                                            'response': 'ممنون از این که به سوالات پاسخ دادید. حال می توانید،به ادامه مکالمه با چتبات بپردازید'})
                    return JsonResponse({'message': message, 'response': current_question})

        # Regular chat processing
        # Predict the emotion and stress levels for the message
        chats = Chat.objects.filter(user=user)
        disorder = check_for_stress_in_text(message, disorder_model, disorder_tokenizer)
        emotion = predict_emotion_label(message, emotion_model, emotion_tokenizer)
        chat = Chat(
            user=user,
            message=message,
            created_at=timezone.now(),
            emotion=emotion,
            disorder=disorder,
            gad_7=False,
            phq_9=False,
        )

        for _ in range(5):
            # Get a response from OpenAI based on the chat history and current message
            response = ask_openai(chat, chat_history=chats, window_size=20)
            validation = predict_validator_labels(response, validator_model, validator_tokenizer)
            if not validation:
                break
            # logger.info(f"Response-text:{response}\nvalidation list:{validation}\n ")

        chat.validation = validation
        chat.response = response
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': Chat.objects.filter(user=user)})


def login(request):
    if request.method == 'POST':
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

        if password1 == password2:
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
