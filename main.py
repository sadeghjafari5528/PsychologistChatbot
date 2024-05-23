import config
from chatbot.chatbot import send_message
from disorder_detector.stress_detector import check_for_stress_in_text
import openai

openai.api_key = config.OPENAI_API_KEY


def main():
    input_text = "سلام وقت‌بخیر"
    check_for_stress_in_text(input_text)
    print(send_message(message=''))


if __name__ == '__main__':
    main()