import openai
import time
# import pandas as pd

def send_message(message: str) -> str:
    try:
        messages = [ {"role": "system", "content":
                "You are a intelligent assistant."} ]

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