import argparse
from tool import *
from sympy import *
from sympy.parsing.latex import parse_latex
import re


def check_latex_equality(latex1, latex2):
    # Convert LaTeX to sympy expressions
    latex1, latex2 = custom_check(latex1, latex2)
    expr1 = sympify(parse_latex(latex1)).evalf()
    expr2 = sympify(parse_latex(latex2)).evalf()
    # Compare the evaluated expressions
    value = abs(expr1 - expr2) <= 1e-2
    return bool(value)


def cleanse_str(strx):
    'Cleanse the string for proper comparison'

    # def replace_pi(expression):
    #     # Replace number\pi with number * 3.1415
    #     expression = re.sub(r'(\d+)\\pi', r'\1 * 3.1415', expression)
    #     # Replace standalone \pi with 3.1415
    #     expression = re.sub(r'\\pi', '3.1415', expression)
    #     return expression

    # if 'pi' in strx:
    #     strx = replace_pi(strx)

    if '^\circ' in strx:
        strx = strx.replace('^\circ', '')

    if '^\\circ' in strx:
        strx = strx.replace('^\\circ', '')

    if '\\text{ degrees}' in strx:
        strx = strx.replace('\\text{ degrees}', '')

    if 'text{degrees}' in strx:
        strx = strx.replace('text{degrees}', '')

    if '\\text{ inches}' in strx:
        strx = strx.replace('\\text{ inches}', '')

    if 'text{inches}' in strx:
        strx = strx.replace('text{inches}', '')

    if '\\text{ cm}' in strx:
        strx = strx.replace('\\text{ cm}', '')

    if 'text{cm}' in strx:
        strx = strx.replace('text{cm}', '')

    if '\%' in strx:
        strx = strx.replace('\%', '')
    return strx


def custom_check(str1, str2):
    cs1 = cleanse_str(str1)
    cs2 = cleanse_str(str2)
    return cs1, cs2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../output/gpt4/geometry_sc5_s0_e455_12_25_21_55.jsonl', type=str,
                        required=False)
    args = parser.parse_args()
    input_path = args.input_path
    output_data = jsonlines_load(input_path)
    annot_path = input_path.split('.jsonl')[0] + '.ajson'
    if os.path.exists(annot_path):
        annot_results = json_load(annot_path)
    else:
        annot_results = {}

    total = 0
    correct = 0
    error = 0
    mistake_list = []
    for i in range(len(output_data)):
        if str(output_data[i]["index"]) in annot_results:
            new_i = str(output_data[i]["index"])
            if bool(annot_results[new_i]):
                correct += 1
            else:
                mistake_list.append(new_i)
            total += 1
            continue

        if output_data[i]['majority_ans'] != "":
            ### Try Latex Match ###
            try:
                are_latex_same = check_latex_equality(output_data[i]['majority_ans'],
                                                      output_data[i]['answer'])
            except Exception:
                print("Needs manual evaluation!")
                print("\n#####################\n")
                print(f"Majority Answer: {output_data[i]['majority_ans']}\n")
                print(f"GT Answer: {output_data[i]['answer']}\n")
                print("\n#####################\n")
                print("Are they equal? Y/N")
                res = input()
                if res.lower().strip() == 'y':
                    are_latex_same = True
                else:
                    are_latex_same = False

                annot_results.update({output_data[i]['index']: are_latex_same})
                with open(annot_path, 'w') as fout:
                    json.dump(annot_results, fout)

            if are_latex_same:
                correct += 1
            else:
                mistake_list.append(i)
        else:
            error += 1
        total += 1

    print(
        f'Accuracy: {(correct) / (total - error)}, Total: {total}, Correct: {correct}, Error: {error}')
    print("##############")
    with open(input_path.split('.jsonl')[0] + '.txt', 'w') as fout:
        for i, j in enumerate(mistake_list):
            fout.write(str(j))
            if i < len(mistake_list) - 1:
                fout.write(',')
