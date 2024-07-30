import random
from random import randrange
import openai
import torch
import gc


def find_difference(new, input_txt):
    min_length = min(len(input_txt), len(new))
    i = 0
    while i < min_length and input_txt[i] == new[i]:
        i += 1
    return new[i:]


def stop_at_specific_tokens(decoded_tokens, stop_tokens):
    stop_positions = []
    for stop_token in stop_tokens:
        if stop_token in decoded_tokens:
            # First occurence #
            stop_positions.append(decoded_tokens.find(stop_token))
    # Cutoff at the first of all tokens #
    cutoff = min(stop_positions) if len(stop_positions) > 0 else len(decoded_tokens) + 1
    return decoded_tokens[:cutoff]


def openai_inference(key, model, max_tokens, stop, messages, temperature, top_p, n):
    result = openai.ChatCompletion.create(
        api_key=key,
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n)
    return result


def huggingface_inference(model, max_tokens, messages, temperature, top_p, n, stop=None, key=None):
    assert isinstance(model, tuple)
    llm, tokenizer = model
    inputs = tokenizer(messages, return_tensors="pt")
    inputs = inputs.to(device='cpu')
    do_sample = True
    max_new_tokens = max_tokens
    top_p = top_p
    temperature = temperature
    num_return_sequences = n
    # Define the tokens you want to stop at
    stop_tokens = ["\n\n", "\n\n\n", "Example", "Math Problem", "Problem"]
    with torch.no_grad():
        outputs = llm.generate(**inputs,
                               max_new_tokens=200,
                               do_sample=do_sample,
                               top_p=top_p,
                               temperature=temperature,
                               num_return_sequences=num_return_sequences)
    ret = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cache_out = [stop_at_specific_tokens(find_difference(ret[i], messages), stop_tokens) for i in
                 range(num_return_sequences)]
    gc.collect()
    torch.cuda.empty_cache()
    return cache_out


def llm_inference(key, model, max_tokens, stop, messages, temperature, top_p, n):
    if isinstance(model, tuple):
        return huggingface_inference(model=model, max_tokens=max_tokens, stop=None, messages=messages,
                                     temperature=temperature, top_p=top_p, n=n, key=None)
    elif isinstance(model, str):
        return openai_inference(model=model, max_tokens=max_tokens, stop=stop, messages=messages,
                                temperature=temperature, top_p=top_p, n=n, key=key)


class FakeLLM:
    def __init__(self, context_size=2048, max_vocab=50_000):
        self.context_size = context_size
        self.max_vocab = max_vocab

    def eval(self):
        return self

    def gen(self, n):
        tensor = torch.LongTensor([randrange(self.max_vocab) for _ in range(100)]).unsqueeze(0)
        if n == 1:
            return tensor
        for i in range(1, n):
            new_tensor = torch.LongTensor([randrange(self.max_vocab) for _ in range(100)]).unsqueeze(0)
            tensor = torch.cat([tensor, new_tensor], dim=0)
        return tensor

    def generate(self, input_ids, **kwargs):
        if input_ids.size()[1] + 300 > self.context_size:
            print(f"WARNING YOUR CURRENT CONTEXT SIZE IS: {input_ids.size()[1]}")
        if 'num_return_sequences' in kwargs:
            n = kwargs['num_return_sequences']
        elif 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 1

        generated_tokens = self.gen(n)

        return torch.cat([input_ids.repeat(n, 1), generated_tokens], dim=1)
