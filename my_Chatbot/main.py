import config
from chatbot.chatbot import get_gpt_message
from disorder_detector.stress_detector import check_for_stress_in_text, load_stress_detector_model_tokenizer

import copy

import openai

from emotion.emotion_detection import load_emotion_detector_model_tokenizer, predict_emotion_label, predict_emotion_of_texts
from message_validator.message_validator import load_validator_model_and_tokenizer, predict_validator_labels

openai.api_key = config.OPENAI_API_KEY


def main():
    k = 1
    stress_tokenizer_detc, stress_model_detc = load_stress_detector_model_tokenizer()
    emotion_model, emotion_tokenizer = load_emotion_detector_model_tokenizer()  

    chat_history = list()

    start_message = 'سلام'
    print(f'Chatbot: {start_message}')
    print('-'*10)
    chat_history.append({"role": "system", "content": start_message})
    while True:
        user_message = input('User: ')
        chat_history.append({"role": "user", "content": user_message})
        #todo: we should check the number of tokens.
        last_k_use_message = [m['content'] for m in chat_history if m['role'] == 'user'][-k:]
        disorder = check_for_stress_in_text('\n'.join(last_k_use_message), stress_model_detc, stress_tokenizer_detc)
        # emotion = predict_emotion_label(last_k_use_message, emotion_model, emotion_tokenizer)
        emotion, _ = predict_emotion_of_texts(last_k_use_message, emotion_model, emotion_tokenizer)
        
        prompt = f'''The emotion of this user is {emotion} and also and hi/she has {disorder} disorder.
          please give him/her a proper message. The message should be short'''
        print(f'Emotion: {emotion}, Disorder: {disorder}')
        status, gpt_message = get_gpt_message(prompt, copy.deepcopy(chat_history))
        if status:
            chat_history.append({"role": "system", "content": gpt_message})
            print(f'Chatbot: {gpt_message}')
            print('-'*10)
        else:
            raise


if __name__ == '__main__':
    main()
    # model, tokenizer = load_validator_model_and_tokenizer()
    # text = "We want to fuck Erfan"
    # labels = predict_validator_labels(text, model, tokenizer)

    # print(f"Predicted Labels: {labels}")