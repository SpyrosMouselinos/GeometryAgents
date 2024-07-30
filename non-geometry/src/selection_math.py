import json
import os
import openai
import random
import time
from datetime import datetime
import argparse
from tqdm import tqdm
from typing import Union
from prompts import math_prompt
from collections import OrderedDict, Counter

from src.selection_geometry import simple_user_assistant_merge
from tool import *


def find_memory_entry(answer_list, solution_list, final_answer):
    indexes = [i for i, x in enumerate(answer_list) if x is not None and abs(x - final_answer) < 1e-3]
    if not indexes:
        return None
    corresponding_strings = [solution_list[i] for i in indexes]
    most_common_string = Counter(corresponding_strings).most_common(1)[0][0]
    return most_common_string


def get_user_assistant_messages(system_message: str, user_message: str, assistant_message: str, memory, type='cot'):
    '''
    This function is used to convert the prompt into the message format used by OpenAI Chat API.
    '''
    messages = []
    messages.append({"role": "system", "content": system_message})
    split_user_messages = user_message.split('\n\n\n\n')
    split_assistant_messages = assistant_message.split('\n\n\n\n')

    for i in range(len(split_user_messages)):
        question = split_user_messages[i]
        answer = split_assistant_messages[i]
        messages += [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{answer}"},
        ]

    if type == 'cot' or type == 'pal':
        for i in range(len(memory)):
            if memory[i] is None or memory[i][0] is None or memory[i][1] is None:
                continue
            else:
                messages += [
                    {"role": "user", "content": f"{memory[i][0]}"},
                    {"role": "assistant", "content": f"{memory[i][1]}"},
                ]
    elif type == 'sel':
        pass
    return messages



def get_cot_prompt(data: dict, backbone: str, memory):
    '''
    This function is used to generate the CoT prompt.
    '''
    if backbone == 'gpt4':
        system_message = math_prompt.GPT4_COT_SYSTEM
        user_message = math_prompt.GPT4_COT_USER
        assistant_message = math_prompt.GPT4_COT_ASSISTANT
        cot_memory = [f[0] for f in memory] if len(memory) > 1 else [(None, None)]
        type = 'cot'
    elif backbone == 'chatgpt':
        system_message = math_prompt.TURBO_COT_SYSTEM
        user_message = math_prompt.TURBO_COT_USER
        assistant_message = math_prompt.TURBO_COT_ASSISTANT
        cot_memory = [f[0] for f in memory] if len(memory) > 1 else [(None, None)]
        type = 'cot'
    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message, cot_memory, type)
    question_message = data['question']
    messages += [{"role": "user", "content": f"Question: {question_message}"}]

    return messages


def get_pal_prompt(data: dict, backbone: str, memory):
    '''
    This function is used to generate the PAL prompt.
    '''
    if backbone == 'gpt4':
        system_message = math_prompt.GPT4_PAL_SYSTEM
        user_message = math_prompt.GPT4_PAL_USER
        assistant_message = math_prompt.GPT4_PAL_ASSISTANT
        messages = get_user_assistant_messages(
            system_message, user_message, assistant_message, [], 'sel')

        question_message = data['question']
        messages += [{"role": "user",
                      "content": f"Question: {question_message}\n\n# solution in Python"}]

    elif backbone == 'chatgpt':
        system_message = math_prompt.TURBO_PAL_SYSTEM
        user_message = math_prompt.TURBO_PAL_USER
        assistant_message = math_prompt.TURBO_PAL_ASSISTANT
        pal_memory = [f[1] for f in memory] if len(memory) > 1 else [(None, None)]
        messages = get_user_assistant_messages(
            system_message, user_message, assistant_message, pal_memory, 'pal')

        question_message = data['question']
        messages += [{"role": "user",
                      "content": f"Answer the following question in Python: {question_message}"}]

    return messages


def get_cot2pal_prompt(data: dict, cot_answer: str):
    '''
    This function is used to generate the COT2PAL prompt.
    '''
    system_message = math_prompt.TURBO_A2P_SYSTEM
    user_message = math_prompt.TURBO_A2P_USER
    assistant_message = math_prompt.TURBO_A2P_ASSISTANT
    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message, [], 'sel')

    question_message = data['question']
    messages += [{"role": "user",
                  "content": f"Convert the following analytical answer to Python: {question_message}\n{cot_answer}\nPython:\n"}]
    return messages


def get_pal2cot_prompt(data: dict, pal_answer: str):
    '''
    This function is used to generate the PAL2COT prompt.
    '''
    system_message = math_prompt.TURBO_P2A_SYSTEM
    user_message = math_prompt.TURBO_P2A_USER
    assistant_message = math_prompt.TURBO_P2A_ASSISTANT
    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message, [], 'sel')

    question_message = data['question']
    messages += [{"role": "user",
                  "content": f"Convert the following Python answer to an analytical step-by-step solution: {question_message}\n{pal_answer}\nAnalytical:\n"}]
    return messages


def get_select_prompt(data: dict, cot_solution: list, pal_solution: list, backbone: str):
    '''
    This function is used to generate the selection prompt.
    '''
    if backbone == 'gpt4':
        system_message = math_prompt.GPT4_SELECT_SYSTEM
        user_message = math_prompt.GPT4_SELECT_USER
        assistant_message = math_prompt.GPT4_SELECT_ASSISTANT
    elif backbone == 'chatgpt':
        system_message = math_prompt.TURBO_SELECT_SYSTEM
        user_message = math_prompt.TURBO_SELECT_USER
        assistant_message = math_prompt.TURBO_SELECT_ASSISTANT
    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message, [], 'sel')

    try:
        pal_generated_list = pal_solution[0].split('"""')
        pal_generated = pal_generated_list[0].strip(
        ) + pal_generated_list[2]
    except Exception as e:
        pal_generated = pal_solution[0]

    if cot_solution[0].startswith('Answer:'):
        cot_generated = cot_solution[0]
    else:
        cot_generated = 'Answer:\n' + cot_solution[0]

    user_message = f'''Math problem: {data['question'].strip()}

(A)
{cot_generated.strip()}

(B)
{pal_generated.strip()}

Which of the above two choices can correctly answer the math problem?'''

    messages += [{"role": "user", "content": user_message}]

    return messages


def query_cot(data: dict, key: str, cot_temperature: float, sc_num: float, backbone: str, memory):
    '''
    This function is used to query OpenAI for CoT solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the CoT solution
    '''

    if backbone == 'gpt4':
        model_name = 'gpt-4'
        query_message = get_cot_prompt(data, backbone=backbone, memory=memory)
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
        query_message = get_cot_prompt(data, backbone=backbone, memory=memory)
    elif backbone == 'mm':
        query_message = get_cot_prompt(data, backbone=backbone, memory=memory)

    start_time = time.time()
    completions = []
    while True:
        try:
            cot_solution = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=500,
                stop='\n\n\n',
                messages=query_message,
                temperature=cot_temperature,
                top_p=1.0,
                n=sc_num)
        except Exception as e:
            print(e)
            cot_solution = None

        if cot_solution is not None:
            completions.extend([choice['message']['content']
                                for choice in cot_solution['choices']])
            # completions = completions[:1]
            return completions
        else:
            sleep_time = 1
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_pal(data: dict, key: str, pal_temperature: float, backbone: str, memory):
    '''
    This function is used to query OpenAI for PAL solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        pal_temperature: the temperature used in PAL
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the PAL solution
    '''
    query_message = get_pal_prompt(data, backbone=backbone, memory=memory)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
    start_time = time.time()
    completions = []
    while True:
        try:
            pal_solution = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=500,
                stop='\n\n\n',
                messages=query_message,
                temperature=pal_temperature,
                top_p=1.0,
                n=1)
        except Exception as e:
            pal_solution = None

        if pal_solution is not None:
            completions.extend([choice['message']['content']
                                for choice in pal_solution['choices']])
            return completions
        else:
            sleep_time = 1
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_cot2pal(data: dict, key: str, backbone: str, cot_answer: str):
    '''
    This function is used to query OpenAI for CoT-->PAL conversion.
    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the PAL solution
    '''
    query_message = get_cot2pal_prompt(data, cot_answer=cot_answer)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'

    start_time = time.time()
    completions = []
    while True:
        try:
            pal_solution = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=500,
                stop='\n\n\n',
                messages=query_message,
                temperature=0.2,
                top_p=0.95,
                n=1)
        except Exception as e:
            pal_solution = None

        if pal_solution is not None:
            completions.extend([choice['message']['content']
                                for choice in pal_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = 1
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_pal2cot(data: dict, key: str, backbone: str, pal_answer: str):
    '''
    This function is used to query OpenAI for PAL--> COT conversion.
    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the COT solution
    '''
    query_message = get_pal2cot_prompt(data, pal_answer=pal_answer)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'

    start_time = time.time()
    completions = []
    while True:
        try:
            cot_solution = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=500,
                stop='\n\n\n',
                messages=query_message,
                temperature=0.2,
                top_p=0.95,
                n=1)
        except Exception as e:
            cot_solution = None

        if cot_solution is not None:
            completions.extend([choice['message']['content']
                                for choice in cot_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = 1
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_selection(data: dict, key: str, cot_solution: list, pal_solution: list, backbone: str):
    '''
    This function is used to query OpenAI for selection solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_solution: a list containing the CoT solution
        pal_solution: a list containing the PAL solution
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the selection solution
    '''
    selection_message = get_select_prompt(
        data, cot_solution, pal_solution, backbone=backbone)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
    start_time = time.time()
    completions = []
    while True:
        try:
            selection_solution = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=200,
                stop='\n\n',
                messages=selection_message,
                temperature=0.,
                top_p=1.0,
                n=1)
        except Exception as e:
            selection_solution = None

        if selection_solution is not None:
            completions.extend([choice['message']['content']
                                for choice in selection_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = 1
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def get_val_prompt(data, proposal):
    '''
    This function is used to generate the validation prompt.
    I don't see why it should be different between GPT4 / ChatGPT
    '''
    system_message = math_prompt.VAL_SYSTEM
    user_message = math_prompt.VAL_USER
    assistant_message = math_prompt.VAL_ASSISTANT

    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message, [], 'sel')
    question_message = data['question']
    messages += [{"role": "user", "content": f"Question: {question_message}\n\nProposed Answer: {proposal}"}]
    return messages


def query_validator(data, key, backbone, proposal):
    '''
    This function is used to query OpenAI for validating COT/PAL solutions.
    '''
    query_message = get_val_prompt(data, proposal=proposal)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'

    start_time = time.time()
    completions = []
    while True:
        try:
            validator_opinion = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=500,
                stop='\n\n\n',
                messages=query_message,
                temperature=0.8,
                top_p=0.95,
                n=5)
        except Exception as e:
            print(e)
            validator_opinion = None

        if validator_opinion is not None:
            completions.extend([choice['message']['content']
                                for choice in validator_opinion['choices']])
            # completions = completions[:1]
            return completions
        else:
            sleep_time = 1
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def get_agreement_prompt(data, proposal, correction):
    '''
    This function is used to generate the agreement prompt.
    I don't see why it should be different between GPT4 / ChatGPT
    '''
    system_message = math_prompt.AGR_SYSTEM
    user_message = math_prompt.AGR_USER
    assistant_message = math_prompt.AGR_ASSISTANT

    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message, [], 'sel')
    question_message = data['question']
    messages += [{"role": "user",
                  "content": f"Question: {question_message}\n\n"
                             f"Proposed Answer: {proposal}\n\n"
                             f"Corrected Answer: {correction}"}]
    return messages


def query_agreement(data, key, mode, backbone, proposal, correction):
    '''
    This function is used to query OpenAI for validating agreement between solutions.
    '''
    query_message = get_agreement_prompt(data, proposal=proposal, correction=correction, backbone=backbone, mode=mode)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'

    start_time = time.time()
    completions = []
    while True:
        try:
            validator_opinion = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=500,
                stop='\n\n\n',
                messages=query_message,
                temperature=0.0,
                top_p=1.0,
                n=1)
        except Exception as e:
            print(e)
            validator_opinion = None

        if validator_opinion is not None:
            completions.extend([choice['message']['content']
                                for choice in validator_opinion['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = 1
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_dialogue(data: dict,
                   key: str,
                   backbone: str,
                   sc_num=1,
                   use_validators=False,
                   memory=[(None, None)]):
    cot_answers = []
    cot_solutions = []
    cot_solution = query_cot(data=data,
                             key=key,
                             cot_temperature=cot_temperature,
                             sc_num=sc_num,
                             backbone=backbone,
                             memory=memory)
    if cot_solution is None:
        print('Time out')
        return None
    else:
        for i in range(len(cot_solution)):
            cot_ans = extract_num_turbo(cot_solution[i])
            cot_answers.append(cot_ans)
            cot_solutions.append(cot_solution[i])

    count = Counter(cot_answers)
    majority_ans = count.most_common(1)[0][0]
    final_solution = None
    for a, s in zip(cot_answers, cot_solutions):
        if a == majority_ans:
            final_solution = s
            break
    if use_validators:
        was_it_correct = []
        corrected_answers = []
        corrected_solutions = []
        cot_validation = query_validator(data=data,
                                         key=key,
                                         backbone=backbone,
                                         proposal=final_solution)
        if cot_validation is None:
            print('Time out')
            return None
        else:
            for i in range(len(cot_validation)):
                vcorrectness, vsolution, vanswer = extract_validity_turbo(cot_validation[i])
                was_it_correct.append(vcorrectness)
                corrected_answers.append(vanswer)
                corrected_solutions.append(vsolution)

        count = Counter(corrected_solutions)
        cot_validation = count.most_common(1)[0][0]
        # Run Agreement
        # cot_agreement = query_agreement(data=data, key=key, mode='cot', backbone=backbone, proposal=cot_result,
        #                                 correction=cot_validation)
    else:
        cot_validation = final_solution
    # Run COT --> PAL
    cot2pal_result = query_cot2pal(data=data,
                                   key=key,
                                   backbone=backbone,
                                   cot_answer=cot_validation)
    use_validators = False
    if use_validators:
        was_it_correct = []
        corrected_answers = []
        corrected_solutions = []
        pal_validation = query_validator(data=data, key=key, backbone=backbone, proposal=cot2pal_result)
        if pal_validation is None:
            print('Time out')
            return None
        else:
            for i in range(len(pal_validation)):
                vcorrectness, vsolution, vanswer = extract_validity_turbo(pal_validation[i])
                was_it_correct.append(vcorrectness)
                corrected_answers.append(vanswer)
                corrected_solutions.append(vsolution)

        count = Counter(corrected_solutions)
        pal_validation = count.most_common(1)[0][0]
        # Run Agreement
        # pal_agreement = query_agreement(data=data, key=key, mode='pal', backbone=backbone, proposal=cot2pal_result,
        #                                 correction=pal_validation)
    else:
        pal_validation = cot2pal_result
    return pal_validation


def query_math(data: dict, key: str, cot_temperature: float, pal_temperature: float, sc_num: int, backbone: str):
    '''
    This function is used to query OpenAI for answers in arithmetic tasks. It contains three steps:
    1. Query CoT for solutions
    2. Query PAL for solutions
    3. Query model selection answers

    Note that we only query selection answers when CoT and PAL answers are different. Otherwise, we directly use CoT or PAL answers.

    We also use majority voting to select the final answer if we have multiple self-consistency samples.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT. 0 for greedy decoding. We set it to 0.5 for self-consistency samples.
        pal_temperature: the temperature used in PAL. 0 for greedy decoding. We set it to 0.8 for self-consistency samples.
        sc_num: the number of self-consistency samples
        backbone: ChatGPT or GPT-4

    Returns:
        to_dump_data: a dict containing the question, answer, the final answer and other information
    '''

    cot_answers = []
    pal_answers = []
    cot_solutions = []
    pal_solutions = []
    selection_solutions = []
    final_answers = []

    for i in range(sc_num):
        cot_ans = None
        pal_ans = None
        selection_ans = None
        final_ans = None
        cot_solution = query_cot(
            data, key, cot_temperature, backbone=backbone, memory=[(None, None)], sc_num=1)
        if cot_solution is None:
            print('Time out')
            return None
        else:
            cot_ans = extract_num_turbo(cot_solution[0])
            cot_answers.append(cot_ans)
            cot_solutions.append(cot_solution[0])

        pal_solution = query_pal(
            data, key, pal_temperature, backbone=backbone, memory=[(None, None)])
        if pal_solution is None:
            print('Time out')
            return None
        else:
            pal_ans = safe_execute_turbo(pal_solution[0])
            pal_answers.append(pal_ans)
            pal_solutions.append(pal_solution[0])

        if cot_ans is not None and pal_ans is not None:

            # ==== Only select when CoT and PAL are different ====
            if abs(cot_ans - pal_ans) > 1e-3:
                selection_ans = query_selection(
                    data, key, cot_solution=cot_solution, pal_solution=pal_solution, backbone=backbone)
                if selection_ans is None:
                    print('Time out')
                    return None
                else:
                    selection_choice = extract_choice_turbo(selection_ans[0])
                    selection_solutions.append(selection_ans[0])
                    if selection_choice == '(A)':
                        final_ans = cot_ans
                    elif selection_choice == '(B)':
                        final_ans = pal_ans
            else:
                final_ans = cot_ans

        elif cot_ans is not None and pal_ans is None:
            final_ans = cot_ans
        elif cot_ans is None and pal_ans is not None:
            final_ans = pal_ans
        else:
            final_ans = None

        final_answers.append(final_ans)

    count = Counter(final_answers)
    majority_ans = count.most_common(1)[0][0]

    # === dump data ===
    to_dump_data = OrderedDict(
        {'index': data['index'], 'question': data['question'], 'answer': data['answer'],
         'majority_ans': majority_ans, 'final_answers': final_answers,
         'cot_executed': cot_answers, 'pal_executed': pal_answers,
         'cot_generated': cot_solutions, 'pal_generated': pal_solutions, 'choice_solution': selection_solutions}
    )

    return to_dump_data


def query_math_memory(data: dict,
                      key: str,
                      cot_temperature: float,
                      pal_temperature: float,
                      sc_num: int,
                      backbone: str,
                      memory=None,
                      ):
    '''
    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT. 0 for greedy decoding. We set it to 0.5 for self-consistency samples.
        pal_temperature: the temperature used in PAL. 0 for greedy decoding. We set it to 0.8 for self-consistency samples.
        sc_num: the number of self-consistency samples
        backbone: ChatGPT or GPT-4
        memory: The memory
    '''
    cot_answers = []
    pal_answers = []
    cot_solutions = []
    pal_solutions = []
    selection_solutions = []
    final_answers = []

    for i in range(sc_num):
        cot_ans = None
        pal_ans = None
        selection_ans = None
        final_ans = None
        cot_solution = query_cot(
            data, key, cot_temperature, sc_num=1, backbone=backbone, memory=memory)
        if cot_solution is None:
            print('Time out')
            return None
        else:
            cot_ans = extract_num_turbo(cot_solution[0])
            cot_answers.append(cot_ans)
            cot_solutions.append(cot_solution[0])

        pal_solution = query_pal(
            data, key, pal_temperature, sc_num=1, backbone=backbone, memory=memory)
        if pal_solution is None:
            print('Time out')
            return None
        else:
            pal_ans = safe_execute_turbo(pal_solution[0])
            pal_answers.append(pal_ans)
            pal_solutions.append(pal_solution[0])

        if cot_ans is not None and pal_ans is not None:

            # ==== Only select when CoT and PAL are different ====
            if abs(cot_ans - pal_ans) > 1e-3:
                selection_ans = query_selection(
                    data, key, cot_solution=cot_solution, pal_solution=pal_solution, backbone=backbone)
                if selection_ans is None:
                    print('Time out')
                    return None
                else:
                    selection_choice = extract_choice_turbo(selection_ans[0])
                    selection_solutions.append(selection_ans[0])
                    if selection_choice == '(A)':
                        final_ans = cot_ans
                    elif selection_choice == '(B)':
                        final_ans = pal_ans
            else:
                final_ans = cot_ans

        elif cot_ans is not None and pal_ans is None:
            final_ans = cot_ans
        elif cot_ans is None and pal_ans is not None:
            final_ans = pal_ans
        else:
            final_ans = None

        final_answers.append(final_ans)

    final_answers.sort()
    count = Counter(final_answers)
    majority_ans = count.most_common(1)[0][0]
    ### Memory Build ###
    if majority_ans is not None and abs(majority_ans - data['answer']) < 1e-3:
        cot_memory = (data['question'], find_memory_entry(cot_answers, cot_solutions, majority_ans))
        pal_memory = (data['question'], find_memory_entry(pal_answers, pal_solutions, majority_ans))
        if cot_memory[1] is None:
            pal2cot = query_pal2cot(data=data, key=key, backbone=backbone, pal_answer=pal_memory[1])
            cot_memory = (data['question'], pal2cot[0])
        if pal_memory[1] is None:
            cot2pal = query_cot2pal(data=data, key=key, backbone=backbone, cot_answer=cot_memory[1])
            pal_memory = (data['question'], cot2pal[0])
        new_memory_entry = [cot_memory, pal_memory]
    else:
        if cot_answers is not None:
            cot_answers.sort()
            cot_count = Counter(cot_answers)
            cot_majority_ans = cot_count.most_common(1)[0][0]
            cot_memory = (data['question'], find_memory_entry(cot_answers, cot_solutions,
                                                              cot_majority_ans) + f"\nThis step-by-step solution is not correct. The correct answer was: {data['answer']} Avoid similar mistakes in the future.")
        else:
            cot_memory = (None, None)
        if pal_answers is not None:
            pal_answers.sort()
            pal_count = Counter(pal_answers)
            pal_majority_ans = pal_count.most_common(1)[0][0]
            pal_memory = (data['question'], find_memory_entry(pal_answers, pal_solutions,
                                                              pal_majority_ans) + f"\nThis Python solution is not correct. The correct answer was: {data['answer']} Avoid similar mistakes in the future.")
        else:
            pal_memory = (None, None)
        new_memory_entry = [cot_memory, pal_memory]

    ####################
    # === dump data ===
    to_dump_data = OrderedDict(
        {'index': data['index'], 'question': data['question'], 'answer': data['answer'],
         'majority_ans': majority_ans, 'final_answers': final_answers,
         'cot_executed': cot_answers, 'pal_executed': pal_answers,
         'cot_generated': cot_solutions, 'pal_generated': pal_solutions, 'choice_solution': selection_solutions}
    )

    return to_dump_data, new_memory_entry


def query_agents_memory(data: dict,
                        key: str,
                        sc_num: int,
                        backbone: str,
                        memory=[(None, None)],
                        use_validators=False
                        ):
    '''
    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT. 0 for greedy decoding. We set it to 0.5 for self-consistency samples.
        pal_temperature: the temperature used in PAL. 0 for greedy decoding. We set it to 0.8 for self-consistency samples.
        sc_num: the number of self-consistency samples
        backbone: ChatGPT or GPT-4
        memory: The memory
    '''
    agent_solution = query_dialogue(data=data,
                                    key=key,
                                    backbone=backbone,
                                    sc_num=sc_num,
                                    use_validators=use_validators,
                                    memory=memory)
    if agent_solution is None:
        print('Time out')
        return None, [(None, None)]
    else:
        agent_ans = safe_execute_turbo(agent_solution[0])
    ### Memory Build ###
    if agent_ans is not None and abs(agent_ans - data['answer']) < 1e-3:
        new_memory_entry = [(None, None)]
        # pal_memory = (data['question'], agent_solution[0])
        # pal2cot = query_pal2cot(data=data, key=key, backbone=backbone, pal_answer=pal_memory[1])
        # pal2cot = pal2cot[0].replace('Analytical', 'Answer')
        # cot_memory = (data['question'], pal2cot)
        # new_memory_entry = [cot_memory, pal_memory]
    else:
        # if cot_answers is not None:
        #     cot_answers.sort()
        #     cot_count = Counter(cot_answers)
        #     cot_majority_ans = cot_count.most_common(1)[0][0]
        #     cot_memory = (data['question'], find_memory_entry(cot_answers, cot_solutions,
        #                                                       cot_majority_ans) + f"\nThis step-by-step solution is not correct. The correct answer was: {data['answer']} Avoid similar mistakes in the future.")
        # else:
        #     cot_memory = (None, None)
        # if pal_answers is not None:
        #     pal_answers.sort()
        #     pal_count = Counter(pal_answers)
        #     pal_majority_ans = pal_count.most_common(1)[0][0]
        #     pal_memory = (data['question'], find_memory_entry(pal_answers, pal_solutions,
        #                                                       pal_majority_ans) + f"\nThis Python solution is not correct. The correct answer was: {data['answer']} Avoid similar mistakes in the future.")
        # else:
        #     pal_memory = (None, None)
        new_memory_entry = [(None, None)]
    ####################
    # === dump data ===
    to_dump_data = OrderedDict(
        {'index': data['index'], 'question': data['question'], 'answer': data['answer'],
         'majority_ans': agent_ans, 'final_answers': None,
         'cot_executed': None, 'pal_executed': None,
         'cot_generated': None, 'pal_generated': None, 'choice_solution': None}
    )

    return to_dump_data, new_memory_entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=650)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--run_only', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=[
        'gsm8k', 'svamp', 'asdiv', 'singleeq', 'singleop',
        'singleaddsub', 'multiarith'], default='gsm8k')
    parser.add_argument('--backbone', type=str,
                        choices=['chatgpt', 'gpt4'], default='gpt4')
    parser.add_argument('--cot_temperature', type=float, default=0.5)
    parser.add_argument('--pal_temperature', type=float, default=0.8)
    parser.add_argument('--sc_num', type=int, default=5,
                        help='Self-consistency samples. 1 indicates greedy decoding')
    parser.add_argument('--output_dir', type=str, default='../output/')
    # TODO: REMOVE
    parser.add_argument(
        '--key', type=str, default='', required=False)

    args = parser.parse_args()

    start_index = args.start
    end_index = args.end
    dataset_name = args.dataset
    cot_temperature = args.cot_temperature
    pal_temperature = args.pal_temperature
    backbone = args.backbone
    run_only = args.run_only
    sc_num = args.sc_num
    output_dir = args.output_dir
    key = args.key

    start_time_0 = time.time()
    print('Current time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))

    dt_string = datetime.now().strftime("%m_%d_%H_%M")

    if dataset_name == 'gsm8k':
        dataset = jsonlines_load('../dataset/gsm8K_test.jsonl')
    elif dataset_name == 'svamp':
        dataset = jsonlines_load('../dataset/svamp.jsonl')
    elif dataset_name == 'asdiv':
        dataset = jsonlines_load('../dataset/asdiv.jsonl')
    elif dataset_name == 'singleeq':
        dataset = jsonlines_load('../dataset/single_eq.jsonl')
    elif dataset_name == 'singleop':
        dataset = jsonlines_load('../dataset/single_op.jsonl')
    elif dataset_name == 'singleaddsub':
        dataset = jsonlines_load('../dataset/single_addsub.jsonl')
    elif dataset_name == 'multiarith':
        dataset = jsonlines_load('../dataset/multiarith.jsonl')

    # === slice data based on start and end === #
    if run_only is None:
        total_num = len(dataset)
        print('total data: ', total_num)
        if end_index == -1:
            end_index = total_num

        if end_index > total_num:
            end_index = total_num

        tasks = dataset[start_index:end_index]
        task_num = len(tasks)
        total_num = task_num
        print('Current total tasks: ', task_num)
        run_only_list = [f for f in range(start_index, end_index)]
    else:
        with open(run_only, 'r') as fin:
            run_only_list = fin.read().strip()
            run_only_list = [int(i) for i in run_only_list.split(',')]
        total_num = len(run_only_list)
        print('total data: ', total_num)
        start_index = 0
        end_index = len(dataset)
        task_num = len(dataset)
        tasks = dataset[start_index:end_index]

    unfinished_tasks = []

    output_path = os.path.join(output_dir, f'{backbone}/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_path = os.path.join(output_path,
                             f'{dataset_name}_sc{sc_num}_s{start_index}_e{end_index}_{dt_string}.jsonl')

    # === run experiments ===
    progress_bar = tqdm(range(total_num))
    memory = [(None, None)]
    for i in range(len(dataset)):
        if i not in run_only_list:
            continue
        task = tasks[i]
        wait_time = min(sc_num * 300, 2000)
        start_time = time.time()
        while True:
            try:
            # ans, new_memory_entry = query_math_memory(
            #     task, key=key, cot_temperature=cot_temperature,
            #     pal_temperature=pal_temperature, sc_num=sc_num, backbone=backbone, memory=memory)
            # memory.append(new_memory_entry)
            # print(f"\n\nROUND: {i}\n\n")
            # print("MEMORY:\n")
            # print(memory)
            #     ans = query_math(
            #         task, key=key, cot_temperature=cot_temperature,
            #         pal_temperature=pal_temperature, sc_num=sc_num, backbone=backbone)
                ans, new_memory_entry = query_agents_memory(task,
                                                            key=key,
                                                            sc_num=sc_num,
                                                            backbone=backbone,
                                                            memory=memory,
                                                            use_validators=True)
                memory.append(new_memory_entry)
            except Exception as e:
                ans = None

            if ans is not None:
                with open(save_path, "a+") as fout:
                    fout.write(json.dumps(ans) + '\n')
                progress_bar.update(1)
                break
            else:
                sleep_time = 1
                time.sleep(sleep_time)

            if time.time() - start_time > wait_time:
                print('Time out')
                print('Current Task: ', i)
                unfinished_tasks.append(task)
                break

        sleep_time = 1
        time.sleep(sleep_time)

    end_time_0 = time.time()
    print('Finish at time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f'Time used: {end_time_0 - start_time_0} seconds')
    if len(unfinished_tasks) > 0:
        print('Unfinished tasks: ')
        for task in unfinished_tasks:
            print(task)

    print('Done')
