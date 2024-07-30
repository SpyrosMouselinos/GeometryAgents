import argparse
from tool import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../output/gpt4/svamp_ms@5.jsonl', type=str,
                        required=False)
    parser.add_argument('--dataset_type', type=str, default='math',
                        choices=['math', 'date'], required=False)
    args = parser.parse_args()

    input_path = args.input_path
    dataset_type = args.dataset_type

    output_data = jsonlines_load(input_path)

    total = 0
    correct = 0
    error = 0
    mistake_list = []
    for i in range(len(output_data)):
        if dataset_type == 'math':
            if output_data[i]['majority_ans'] is not None:
                if abs(output_data[i]['majority_ans'] - output_data[i]['answer']) < 1e-3:
                    correct += 1
                else:
                    mistake_list.append(i)
            else:
                error += 1

        else:
            if output_data[i]['final_ans'] == output_data[i]['answer']:
                correct += 1
            else:
                mistake_list.append(i)
                error += 1

        total += 1

    print(
        f'Accuracy: {(correct ) / (total - error)}, Total: {total}, Correct: {correct}, Error: {error}')
    print("##############")
    with open(input_path.split('.jsonl')[0] + '.txt', 'w') as fout:
        for i, j in enumerate(mistake_list):
            fout.write(str(j))
            if i < len(mistake_list) - 1:
                fout.write(',')
