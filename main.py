import config
from chatbot.chatbot import send_message

import openai

openai.api_key = config.OPENAI_API_KEY

def main():
    print(send_message(text=''))

if __name__ == '__main__':
    main()