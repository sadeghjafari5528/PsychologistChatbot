import openai
import time
# import pandas as pd

def get_gpt_message(message: str, history: list) -> str:
    try:
        # history = [ {"role": "system", "content":
        #         "You are a intelligent assistant."} ]

        history.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613", messages=history
        )
        reply = chat.choices[0].message.content
        return True, reply
    except Exception as e:
        return False, e