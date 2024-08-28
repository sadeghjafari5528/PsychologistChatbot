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

from chatbot.disorder_detector.stress_detector import check_for_stress_in_text, load_stress_detector_model_tokenizer
from chatbot.emotion.emotion_detection import load_emotion_detector_model_tokenizer, predict_emotion_label, \
    predict_emotion_of_texts, label_dict
from chatbot.message_validator.message_validator import load_validator_model_and_tokenizer, predict_validator_labels

from dotenv import load_dotenv
import os

load_dotenv()

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
به من در تلگرام با ای دی  sadeghjafari1379 پیام بدید یا در لینکدین با من در ارتباط باشید.

💢فقط یه نکته ۴ روز بیشتر فرصت ندارین تا تصمیمتون رو به ما اعلام کنید.
"""

openai.api_key = os.getenv('OPENAI_API_KEY')

validator_model, validator_tokenizer = load_validator_model_and_tokenizer()
emotion_model, emotion_tokenizer = load_emotion_detector_model_tokenizer()
disorder_tokenizer, disorder_model = load_stress_detector_model_tokenizer()

# logger = logging.getLogger('django')


def calculate_weighted_average(chats: list[Chat], feature: str, decay_factor: float = 0.9):
    weighted_average = dict()

    for label in getattr(chats[0], feature).keys():
        total_weight = 0
        weighted_scores = list()
        for chat in chats:
            if chat.response == cooperation_text:
                continue
            day_diff = (timezone.now() - chat.created_at).days
            weight = np.exp(-decay_factor * day_diff)
            weighted_scores.append(getattr(chat, feature)[label] * weight)
            total_weight += weight

        weighted_average[label] = sum(weighted_scores) / total_weight if total_weight != 0 else 0
    return weighted_average


def ask_openai(chat_obj: Chat, chat_history, window_size: int = None):
    chat_history_size = len(chat_history)
    if window_size:
        chat_history = chat_history.order_by('-id')[:window_size]

    messages = list()
    for chat in chat_history:
        messages.append({"role": "user", "content": chat.message})
        messages.append({"role": "system", "content": chat.response})

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
"""
    else:
        prompt = f"""
The previous messages are the chat history between a patient and a psychologist.
Suppose you are a professional psychologist. Based on the following information,
respond to the patient with a short message.(Prevent to say 'Hi' in each message. And only speak in persian)

Patient message: {chat_obj.message}
"""

    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
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
    return [
        "در طول دو هفته گذشته، چند بار احساس کرده‌اید که علاقه یا لذتی به فعالیت‌هایی که معمولاً از آنها لذت می‌بردید، ندارید؟     \n     عدد گزینه را واردکنید :                  گزینه 0) هرگز   \t    گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز   ",
        "در طول دو هفته گذشته، چند بار احساس کرده‌اید که افسرده، غمگین یا ناامید هستید       \n     عدد گزینه را واردکنید :                  گزینه  0) هرگز   \t    گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز    ",
        "در طول دو هفته گذشته، چند بار با مشکل در خوابیدن، خواب زیاد یا بیدار شدن در نیمه‌شب مواجه بوده‌اید؟     \n     عدد گزینه را واردکنید :         گزینه 0) هرگز  \t     گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز    ",
        "در طول دو هفته گذشته، چند بار احساس خستگی یا کمبود انرژی داشته‌اید؟     \n     عدد گزینه را واردکنید :    گزینه 0) هرگز \t      گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز    ",
        "در طول دو هفته گذشته، چند بار اشتهای شما کاهش یا افزایش یافته است؟     \n عدد گزینه را واردکنید :   گزینه 0) هرگز        \t       گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز   ",
        "در طول دو هفته گذشته، چند بار احساس کرده‌اید که خود را یک شکست‌خورده می‌دانید یا احساس کرده‌اید که خود یا خانواده‌تان را ناامید کرده‌اید؟       \n     عدد گزینه را واردکنید :           گزینه 0) هرگز  \t     گزینه 1) چندروز             گزینه 2) بیش از نصف روز ها            گزینه  3) تقریبا هرروز    ",
        "در طول دو هفته گذشته، چند بار با مشکل تمرکز کردن بر روی چیزهایی مثل خواندن روزنامه یا تماشای تلویزیون مواجه بوده‌اید؟ \n    عدد گزینه را واردکنید :  گزینه 0) هرگز  \t  گزینه 1) چندروز    گزینه 2) بیش از نصف روزها    گزینه  3) تقریبا هرروز  ",
        "در طول دو هفته گذشته، چند بار حرکت یا صحبت شما به قدری کند بوده است که دیگران متوجه آن شده‌اند؟ یا برعکس، چند بار به قدری بی‌قرار بوده‌اید که نمی‌توانستید آرام بنشینید؟   \n   عدد گزینه را واردکنید :    گزینه 0) هرگز   \t    گزینه 1) چندروز             گزینه 2) بیش از نصف روزها      گزینه  3) تقریبا هرروز        ",
        " در طول دو هفته گذشته، چند بار احساس کرده‌اید که بهتر است خودتان را آسیب بزنید یا خودکشی کنید؟   \nعدد گزینه را واردکنید :                  گزینه 0) هرگز  \t     گزینه 1) چندروز         گزینه 2) بیش از نصف روزها         گزینه  3) تقریبا هرروز   ",
    ]


def chatbot(request):
    user = request.user
    if not user.is_authenticated:
        return redirect('login')

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
    question_record, created = Question.objects.get_or_create(user=user, created_at__date=today)

    if request.method == 'POST':
        message = request.POST.get('message')
        if user.is_staff:
            if question_record.gad_7_count == 0 and not question_record.gad_7_completed and not question_record.phq_9_completed:
                first_question = gad_7_questions()[0]
                Questionnaire.objects.create(
                    user=user,
                    created_at=timezone.now(),
                    question=first_question,
                    answer='',
                    gad_7_number=1,
                    is_gad_7=True,
                    is_phq_9=False
                )
                question_record.gad_7_count += 1
                question_record.save()
                return JsonResponse({'response': first_question})

            if (question_record.gad_7_count > 0 and not question_record.gad_7_completed) or (
                    question_record.gad_7_count == 7 and question_record.phq_9_count == 0):
                try:
                    last_questionnaire_entry = Questionnaire.objects.filter(
                        user=user,
                        gad_7_number=question_record.gad_7_count,
                        is_gad_7=True,
                        is_phq_9=False
                    ).order_by('created_at').last()


                    if last_questionnaire_entry.answer == '':
                        last_questionnaire_entry.answer = message
                        last_questionnaire_entry.save()
                except Questionnaire.DoesNotExist:
                    pass

            elif question_record.phq_9_count > 0 and not question_record.phq_9_completed:
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

            if not question_record.gad_7_completed:
                questions = gad_7_questions()
                if question_record.gad_7_count <= len(questions):
                    if question_record.gad_7_count < len(questions):
                        current_question = questions[question_record.gad_7_count]

                        Questionnaire.objects.create(
                            user=user,
                            created_at=timezone.now(),
                            question=current_question,
                            answer='',
                            gad_7_number=question_record.gad_7_count + 1,
                            is_gad_7=True,
                            is_phq_9=False
                        )
                    if question_record.gad_7_count < 7:
                        question_record.gad_7_count += 1
                        question_record.save()

                    last_questionnaire_entry = Questionnaire.objects.filter(
                        user=user,
                        gad_7_number=question_record.gad_7_count,
                        is_gad_7=True,
                        is_phq_9=False
                    ).order_by('created_at').last()

                    if question_record.gad_7_count == len(questions) and not (last_questionnaire_entry.answer == ''):

                        question_record.gad_7_completed = True
                        question_record.save()
                        return JsonResponse({'message': message,
                                            'response': 'ممنون از این که به سری اول سوالات پاسخ دادید. حال ممنون می شویم با پاسخ به سوالات مربوط به پرسشنامه سری دوم ما را یاری نمایید. '})
                    return JsonResponse({'message': message, 'response': current_question})

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
                                            'response': 'با سپاس فراوان از پاسخگوییتون به سوالات . حال می توانید به ادامه چت با بات بپردازید.'})

                    return JsonResponse({'message': message, 'response': current_question})

        # Regular chat processing
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
