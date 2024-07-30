import json
import os.path
import re
import regex
import func_timeout
from typing import Union
import random
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import prompts.math_prompt
import prompts.geometry_static_prompt
import prompts.geometry_static_prompt_short
from model_inference import FakeLLM


def json_load(fname: str):
    with open(fname, 'r') as f:
        return json.load(f)


def jsonlines_load(fname: str):
    with open(fname, 'r') as f:
        return [json.loads(line) for line in f]


def jsonlines_dump(fname: str, data: Union[dict, list]):
    try:
        with open(fname, 'a+') as f:
            if isinstance(data, dict):
                f.write(json.dumps(data) + '\n')
            elif isinstance(data, list):
                for d in data:
                    f.write(json.dumps(d) + '\n')

    except (FileNotFoundError, FileExistsError) as e:
        print(f'Error: {e}')
        print(f'Could not write to {fname}')


def jsonfolder_load(foname: str):
    if os.path.isdir(foname):
        json_files_in_folder = os.listdir(foname)
        loader = []
        for file_ in json_files_in_folder:
            with open(foname + '/' + file_, 'r') as f:
                content = json.load(f)
            loader.append(content)
    else:
        raise FileNotFoundError(f"Folder {foname} was not found!")
    return loader


def extract_num_codex(solution: str):
    answer = None
    try:
        solution = solution.strip()
        answer = re.findall(
            r'The answer is \$?(-?\d+\,*\.?\d*)\%?', solution)

        if len(answer) == 0:
            answer = re.findall(r'-?\d+\,*\.?\d*', solution)
            answer = answer[-1]
        else:
            answer = answer[0]

        answer = answer.replace(',', '')
        answer = float(answer)

    except Exception as e:
        answer = None

    return answer


def safe_execute_codex(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is not None:
                return [locals_.get(k, None) for k in keys]

            solution = locals_.get('solution', None)
            if solution is not None:
                return solution()
            else:
                exec('\n'.join([xx[4:]
                                for xx in x.strip().split('\n')[1:-1]]))
                locals_ = locals()
                return locals_.get('result', None)

        except Exception:
            return None

    try:
        ans = func_timeout.func_timeout(3, execute, args=(code_string,))
        ans = float(ans) if ans is not None else ans
    except func_timeout.FunctionTimedOut:
        ans = None
    return ans


def extract_num_turbo(solution: str):
    ans = solution.strip().split('\n')[-1].replace('So the answer is ', '')
    prd = [x[0] for x in regex.finditer(
        r'[\d\.,]+', ans) if regex.search(r'\d', x[0])]
    if len(prd) > 2:
        prd = prd[-1]
    elif len(prd):
        prd = prd[0]
    else:
        prd = None
    try:
        prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
    except:
        prd = None
    return prd


def safe_execute_turbo(code_string: str, keys=None):
    def execute(x, code_return):
        try:
            exec(x)
            locals_ = locals()
            if keys is not None:
                return [locals_.get(k, None) for k in keys]

            solution = locals_.get('solution', None)
            if solution is not None:
                return solution()
            else:
                executed_code = 'import math\n' + 'import datetime\n' + \
                                '\n'.join([xx[4:]
                                           for xx in x.strip().split('\n')[1:-1]])
                exec(executed_code)
                locals_ = locals()
                return locals_.get(code_return, None)

        except Exception as exp:
            print('Executing code error', exp)
            return None

    # === find code snippets between def solution(): and return ===
    try:
        code_list = code_string.strip().split('\n')

        new_code_list = []
        all_codes = []
        code_return = 'ans'

        for i in range(len(code_list)):
            if code_list[i].strip() == 'def solution():':
                new_code_list.append(code_list[i])
                for j in range(i + 1, len(code_list)):
                    if code_list[j].startswith('    '):
                        new_code_list.append(code_list[j])
                    if 'return ' in code_list[j]:
                        code_return = code_list[j].split('return ')[1].strip()
                all_codes.append('\n'.join(new_code_list))
                new_code_list = []
        new_code = all_codes[-1]

        ans = func_timeout.func_timeout(
            3, execute, args=(new_code, code_return,))
        ans = ans if ans is not None else ans
    except func_timeout.FunctionTimedOut:
        ans = None

    try:
        ans = float(ans) if ans is not None else ans
    except:
        ans = None

    return ans


def extract_choice_turbo(selection: str):
    if selection.startswith('Both') or selection.startswith('Neither'):
        if random.random() < 0.5:
            choices_a_b = '(A)'
        else:
            choices_a_b = '(B)'
    else:
        try:
            choices = re.findall(r'(\(A\)|\(B\)) can correctly', selection)

            if len(choices) == 0:
                choices = re.findall(
                    r'(\(A\)|\(B\)) is(?:\sthe)? correct', selection)
            choices_a_b = choices[0]

        except:
            if random.random() < 0.5:
                choices_a_b = '(A)'
            else:
                choices_a_b = '(B)'

    return choices_a_b


def extract_validity_turbo(solution: str, keys=None):
    try:
        ans_line_pre = solution.strip().split('Here it is again:')[0]
        ans_line_post = solution.strip().split('Here it is again:')[1][1:]
    except:
        return False, '', ''
    if 'mistake' in ans_line_pre or 'correction' in ans_line_pre:
        if 'def' in solution:
            return False, ans_line_post, safe_execute_turbo(solution, keys)
        else:
            return False, ans_line_post, extract_num_turbo(solution)
    else:
        if 'def' in solution:
            return True, ans_line_post, safe_execute_turbo(solution, keys)
        else:
            return True, ans_line_post, extract_num_turbo(solution)


def extract_choice_codex(selection: str):
    try:
        choice = re.findall(r'\(A\)|\(B\)', selection)

        if choice[0] == '(A)':
            return '(A)'
        elif choice[0] == '(B)':
            return '(B)'
        else:
            if random.random() < 0.5:
                return '(A)'
            else:
                return '(B)'
    except Exception as e:
        if random.random() < 0.5:
            return '(A)'
        else:
            return '(B)'


def extract_date_cot(solution: str):
    cot_answer = None
    cot_answers = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', solution)
    try:
        cot_answer = cot_answers[-1]
        if len(cot_answer.split('/')[0]) == 1:
            cot_answer = '0' + cot_answer

        if len(cot_answer.split('/')[1]) == 1:
            cot_answer = cot_answer.split(
                '/')[0] + '/0' + cot_answer.split('/')[1] + '/' + cot_answer.split('/')[2]
    except Exception as e:
        cot_answer = None

    return cot_answer


def execute_date_pal(code_string: str, keys=None):
    def execute(x, code_return):
        try:
            exec(x)
            locals_ = locals()
            if keys is not None:
                return [locals_.get(k, None) for k in keys]

            solution = locals_.get('solution', None)
            if solution is not None:
                return solution()
            else:
                executed_code = 'import math\n' + 'import datetime\n' + 'from datetime import datetime\n' + 'from dateutil.relativedelta import relativedelta as relativedelta\n' + \
                                'from dateutil.relativedelta import relativedelta as timedelta\n' + \
                                'import pytz\n' + \
                                '\n'.join([xx[4:] for xx in x.strip().split('\n')[1:-1]])
                exec(executed_code)
                locals_ = locals()
                return locals_.get(code_return, None)

        except Exception as exp:
            print('Executing code error', exp)
            return None

    # find code snippets between def solution(): and return
    try:
        code_list = code_string.strip().split('\n')

        new_code_list = []
        all_codes = []
        code_return = 'ans'

        for i in range(len(code_list)):
            if code_list[i].strip() == 'def solution():':
                new_code_list.append(code_list[i])
                for j in range(i + 1, len(code_list)):
                    if code_list[j].startswith('    '):
                        new_code_list.append(code_list[j])
                    if 'return ' in code_list[j]:
                        code_return = code_list[j].split('return ')[1].strip()
                all_codes.append('\n'.join(new_code_list))
                new_code_list = []
        new_code = all_codes[-1]
        ans = func_timeout.func_timeout(
            3, execute, args=(new_code, code_return,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans


def extract_geom(ans_string: str):
    pattern = r'\\boxed{((?:[^{}]|{[^{}]*})+)}'
    answer_match = re.search(pattern, ans_string)
    if answer_match:
        # Extract just the content inside \boxed{}
        answer = answer_match.group(1)
    else:
        answer = ans_string.split('\n')[-1]
    return answer


def load_hf_model(device='fake'):
    # model_path = "akjindal53244/Arithmo-Mistral-7B"
    model_path = "meta-math/MetaMath-Mistral-7B"
    #model_path = "microsoft/phi-1_5"
    run_model_on_gpu = device == 'cuda'
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if device == 'fake':
        model = FakeLLM(context_size=2048, max_vocab=tokenizer.vocab_size - 1)
    else:
        if run_model_on_gpu:
            device_map = {"": 0}
            # Load tokenizer and model with QLoRA configuration
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
            model = model.eval()
        else:
            device_map = {"": "cpu"}
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=True
            )
    return model, tokenizer


if __name__ == '__main__':
    m, t = load_hf_model('fake')
    messages = prompts.geometry_static_prompt.VAL_GEOM_USER
    inputs = t(messages, return_tensors="pt")
    outputs = m.generate(**inputs,
                           max_new_tokens=1,
                           do_sample=1,
                           top_p=1,
                           temperature=1,
                           num_return_sequences=5)
    ret = t.batch_decode(outputs, skip_special_tokens=True)
    #print(ret)