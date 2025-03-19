import base64
import requests
import json
import http.client
import pygame
import io
import subprocess
import os
from past.builtins import raw_input
from graspness import infer_vis_grasp


# global_variables
headers = {
    'Authorization': 'Your gpt api key',
    'Content-Type': 'application/json'
}


# image_coding
def encoder_image(image):
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# text_to_speech
def text2audio(input_text):
    conn = http.client.HTTPSConnection("api.chatanywhere.tech")
    payload = json.dumps({
        "model": "tts-1",
        "input": input_text,
        "voice": "alloy"
    })
    conn.request("POST", "/v1/audio/speech", payload, headers)
    res = conn.getresponse()
    pygame.mixer.init()
    audio_data = io.BytesIO(res.read())
    pygame.mixer.music.load(audio_data, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


# request_gpt
def request_api(message_text):
    url = "XXXXXXXXXXXXXXXXXXXXXXX"
    payload = json.dumps({
       "model": "gpt-4o",
       "messages": message_text
    })
    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.json()['choices'][0]['message']['content']
    return res


# Call demo.py in vlpart to split the mask
def call_vlpart(custom_vocabulary, image_path):
    result = subprocess.run(['python', '../VLPart/demo/demo.py',
                             '--input', image_path,
                             '--custom_vocabulary', custom_vocabulary,
                             ], cwd='/VLPart')

def call_graspnet(seg_label):
    original_cwd = os.getcwd()
    try:
        os.chdir('/graspness')
        translation, rotation, score = infer_vis_grasp.main(seg_label)
        return translation, rotation, score
    finally:
        # Restoring cwd
        os.chdir(original_cwd)


def gpt_infer():
    # Initialization prompt
    prompt = '1.You are a robotic arm. Based on the current image and my command, provide the item I need. If the items in the image do not satisfy my command, please provide an explanation.'

    image_path = 'VLPart_ws/data/input/test_color1.png'
    base64_image = encoder_image(image_path)
    print("*****************************************************")
    print("***************  The dialogue start.  ​***************")
    input1 = raw_input("input：")
    conversation = [{
            "role": "system",
            "content": prompt
        }, {
            "role": "user",
            "content": [{
                  "type": "text",
                  "text": input1
               }, {
                  "type": "image_url",
                  "image_url": {
                     "url": f"data:image/jpeg;base64,{base64_image}"}
               }
            ]
        }
    ]
    answer = request_api(conversation)
    print(f"GPT_Answer: {answer}")
    gpt_output = {"role": "assistant", "content": answer}
    conversation.append(gpt_output)
    # text2audio(answer)
    while True:
        user_input = input("input：")
        if user_input == 'q':
            print("****************** Ending the conversation ​******************")
            print("*************************************************************")
            break
        conversation.append({"role": "user", "content": user_input})
        answer = request_api(conversation)
        print(f"GPT_Answer: {answer}")
        # text2audio(answer)
        conversation.append({"role": "assistant", "content": answer})

    if '-' in answer:
        target_area = answer.replace("-", " ")
    else:
        target_area = answer

    call_vlpart(target_area, image_path)
    # Call the fetching module
    translation, rotation, score = call_graspnet(target_area)
    return target_area, translation, rotation, score


def main():
    target_area, translation, rotation, score = gpt_infer()
    print(f"Target Area: {target_area}")
    print(f"Translation: {translation}")
    print(f"Rotation: {rotation}")
    print(f"Score: {score}")


if __name__ == "__main__":
    main()
