import openai
import time
# import pandas as pd

def send_message(text: str) -> str:
    try:
        messages = [ {"role": "system", "content":
                "You are a intelligent assistant."} ]
        message = f"در متن زیر کدام یک از احساسات روبه‌رو وجود دارد؟ 1-تعجب 2-ناراحتی 3-تنفر 4-خوشحالی 5-ترس 6- عصبانیت 7- هیچکدام\n{text}"
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613", messages=messages
        )
        reply = chat.choices[0].message.content
        return True, reply
    except Exception as e:
        return False, e