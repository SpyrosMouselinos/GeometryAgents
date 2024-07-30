import openai
import os
import pandas as pd
import time
import base64
import requests
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import json
import PIL.Image

from prompt.constants import INCEPTION_PROMPT
from prompt.promptbuilder import few_shot_builder
import argparse

PASS_AT_K = 50

# Set your Open API
openai_key = ''
openai_model_name = ''

## Add your GCP project and location data
project_id = ''
location = ''

vertexai.init(project=project_id, location=location)

gemini_model_name = 'gemini-pro-vision'
gemini_model = GenerativeModel(gemini_model_name)  # For Gemini


# Function to encode the image for OPENAI
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Function to encode the image for Gemini
def encode_image_gemini(image_path):
    return PIL.Image.open(image_path)


def OpenAI_TXT_Query(agent_role, prompt, pass_at_k=1):
    messages = [
        {"role": "assistant", "content": f"{agent_role}"},
        {"role": "user", "content": f"{prompt}"},
    ]
    result = openai.ChatCompletion.create(
        api_key=openai_key,
        model=openai_model_name,
        max_tokens=300,
        stop=['\n\n'],
        messages=messages,
        temperature=0.2,
        top_p=0.95,
        n=pass_at_k)
    return [choice['message']['content'] for choice in result['choices']]


def Gemini_TXT_Query(agent_role, prompt, pass_at_k=1):
    results = []
    for i in range(pass_at_k):
        while True:
            try:
                response = gemini_model.generate_content(
                    agent_role + '\n\n' + prompt,
                    generation_config={
                        'temperature': 0.8,
                        'top_p': 0.95
                    },
                )
                results.append(response.text)
            except Exception as e:
                print(e)
                print("Waiting for vertex...")
                time.sleep(5)

    if len(results) != pass_at_k:
        print(f"You requested {pass_at_k} responses but you got {len(results)}...")
    return results


def OpenAI_VRP_Query(prompt, img):
    base64_image = encode_image(img)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "{prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.95,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return [f['message']['content'] for f in response.json()['choices']][0]


def Gemini_VRP_Query(prompt, img):
    while True:
        try:
            response = gemini_model.generate_content(
                [img, prompt],
                generation_config={
                    'temperature': 0.2,
                    'top_p': 0.95
                },
            )
            break
        except Exception as e:
            print(e)
            print("Waiting for vertex...")
            time.sleep(5)
    return response.text


def Inject_VRP(vrp, question):
    def inject(vrp, subquestion):
        pre = subquestion.split('\n')[0]
        post = subquestion.split('\n')[1]
        return pre + '\n' + vrp + '\n' + post

    return question[:question.rfind('Description:')] + inject(vrp, question[question.rfind('Description:'):])


def process_without_vrp(args, data, alpha_image_list, beta_image_list):
    PASS_AT_K = 50

    flat_data = [row.to_dict() for _, row in data.iterrows()]

    results = {'results': []}
    for problem in flat_data:
        pack = problem['pack']
        question = problem['question']
        few_shot_prompt = few_shot_builder(current_question=question,
                                           dataset_df=data,
                                           pack_limit=pack,
                                           mode='Adapt',
                                           self_reflect=False)
        if args.model == 'gpt4':
            response = OpenAI_TXT_Query(INCEPTION_PROMPT, few_shot_prompt, PASS_AT_K)
        elif args.model == 'gemini':
            response = Gemini_TXT_Query(INCEPTION_PROMPT, few_shot_prompt, PASS_AT_K)
        results['results'].append(response)
        time.sleep(1)

    with open(f'results_no_vrp.json', 'w') as fout:
        json.dump(results, fout)


def process_with_vrp(args, data, alpha_image_list, beta_image_list):
    vrps = []
    for encoded_image in alpha_image_list:
        if args.model == 'gpt4':
            vrps.append(OpenAI_VRP_Query(INCEPTION_PROMPT, encoded_image))
        elif args.model == 'gemini':
            vrps.append(Gemini_VRP_Query(INCEPTION_PROMPT, encoded_image))

    for encoded_image in beta_image_list:
        if args.model == 'gpt4':
            vrps.append(OpenAI_VRP_Query(INCEPTION_PROMPT, encoded_image))
        elif args.model == 'gemini':
            vrps.append(Gemini_VRP_Query(INCEPTION_PROMPT, encoded_image))

    flat_data = [row.to_dict() for _, row in data.iterrows()]

    results = {'results': []}
    for problem, vrp in zip(flat_data, vrps):
        pack = problem['pack']
        question = problem['question']
        few_shot_prompt = few_shot_builder(current_question=question,
                                           dataset_df=data,
                                           pack_limit=pack,
                                           mode='Adapt',
                                           self_reflect=False)
        if args.model == 'gpt4':
            response = OpenAI_TXT_Query(INCEPTION_PROMPT,
                                        Inject_VRP(vrp, few_shot_prompt),
                                        PASS_AT_K)
        elif args.model == 'gemini':
            response = Gemini_TXT_Query(INCEPTION_PROMPT,
                                        Inject_VRP(vrp, few_shot_prompt),
                                        PASS_AT_K)
        time.sleep(1)
        results['results'].append(response)

    with open(f'results_vrp.json', 'w') as fout:
        json.dump(results, fout)


def main(args):
    IMAGE_PATH = './minimal_demo/Images'
    ALPHA_IMAGES = IMAGE_PATH + '/Alpha'
    BETA_IMAGES = IMAGE_PATH + '/Beta'

    print("You are using PIL encoding for images, if it does not work try with Base64 by uncommenting this.")
    ### OpenAI - Use Base64 for image encoding ###
    if args.model == 'gpt4':
        alpha_image_list = [encode_image(ALPHA_IMAGES + '/' + f) for f in os.listdir(ALPHA_IMAGES)]
        beta_image_list = [encode_image(BETA_IMAGES + '/' + f) for f in os.listdir(BETA_IMAGES)]
    ### Gemini - Use PIL for image encoding ###
    elif args.model == 'gemini':
        alpha_image_list = [encode_image_gemini(ALPHA_IMAGES + '/' + f) for f in os.listdir(ALPHA_IMAGES)]
        beta_image_list = [encode_image_gemini(BETA_IMAGES + '/' + f) for f in os.listdir(BETA_IMAGES)]

    data = pd.read_csv('./minimal_demo/filtered_euclidea.csv')
    data = data[(data['pack'] == 'Alpha') | (data['pack'] == 'Beta')]

    if args.use_vrp:
        process_with_vrp(args, data, alpha_image_list, beta_image_list)
    else:
        process_without_vrp(args, data, alpha_image_list, beta_image_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-4 or Gemini model with or without VRP.')
    parser.add_argument('model', choices=['gpt4', 'gemini'], help='Choose which model to run: gpt4 or gemini')
    parser.add_argument('--use_vrp', action='store_true', help='Use VRP in the process')
    args = parser.parse_args()
    main(args)
