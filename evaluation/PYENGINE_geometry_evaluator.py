import copy
import json
import os.path
import numpy as np
import itertools
import pandas as pd
from evaluation.baselines import LargestCommonPrefix, RandomRollout
from evaluation.overlap import tool_overlap



def subsol(series_ws, series):
    series_ws = series_ws.values
    series = series.values
    results = []
    for x, y in zip(series_ws, series):
        results.append([x[len(y):]])
    return results


def populate(somelist, n=20):
    new_list = []
    for i in somelist:
        new_list.append([i] * n)
    return new_list


def estimate_pass_at_k(
        num_samples,
        num_correct,
        k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def find_index_ranges(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Dictionary to store the ranges
    ranges = {}

    # Initial variables
    prev_value = None
    start_index = None

    for i, row in df.iterrows():
        # Check for change in category
        if row[column_name] != prev_value:
            # If not the first category, save the previous category's range
            if prev_value is not None:
                ranges[prev_value] = [start_index, i - 1]

            # Update for new category
            prev_value = row[column_name]
            start_index = i

    # Add the last category's range
    if prev_value is not None:
        ranges[prev_value] = [start_index, df.index[-1]]

    return ranges


def match_index_with_pack(i, reverse_pack_dict):
    for v, k in reverse_pack_dict.items():
        real_v = eval(v)
        if i in range(real_v[0], real_v[1] + 1):
            return k


def match_index_with_level(i, reverse_level_dict):
    for v, k in reverse_level_dict.items():
        real_v = eval(v)
        if i in range(real_v[0], real_v[1] + 1):
            return k


class GeometryEvaluator:

    def __init__(self, dataset, results=None, results_tools=None, results_symbols=None):
        """
        Initializes the GeometryEvaluator with a dataset and model.

        Args:
        - dataset: The levels (CSV) - A list of strings taken from the dataset.
        - results: A list of generated responses (Baselines + DEBUG)
        - results_tools: A list of only tools used from responses (Baselines + DEBUG)
        - results_symbols: A list of only symbols produced from responses (Baselines + DEBUG)
        """
        self.dataset = dataset

        if isinstance(results, np.ndarray):
            results = list(results)
        if isinstance(results_tools, np.ndarray):
            results_tools = list(results_tools)
        if isinstance(results_symbols, np.ndarray):
            results_symbols = list(results_symbols)
        if isinstance(results, list):
            self.results = results
            assert results_tools is not None
            assert results_symbols is not None
            self.results_tools = results_tools
            self.results_symbols = results_symbols
        elif isinstance(results, pd.DataFrame):
            self.results = results['solution']
            self.results_tools = results['solution_tool']
            self.results_symbols = results['solution_symbol']
        else:
            print("Results not in debug format. Skipping....")
            return

        # Fix Errors - Case 1 #
        if isinstance(self.results[0], str):
            self.results = [[f] for f in self.results]

        if isinstance(self.results_tools[0], str):
            self.results_tools = [[eval(f)] for f in self.results_tools]

        if isinstance(self.results_symbols[0], str):
            self.results_symbols = [[eval(f)] for f in self.results_symbols]

        # Fix Errors - Case 2 #
        if isinstance(self.results[0], list) and isinstance(self.results[0][0], str):
            pass

        if isinstance(self.results_tools[0], list) and isinstance(self.results_tools[0][0], str):
            final_result = []
            for k in self.results_tools:
                instance_result = []
                for f in k:
                    instance_result.append(eval(f))
                final_result.append(instance_result)
            self.results_tools = final_result

        # Fix Errors - Case 3 #
        if isinstance(self.results_symbols[0], list) and isinstance(self.results_symbols[0][0], str):
            final_result = []
            for k in self.results_symbols:
                instance_result = []
                for f in k:
                    instance_result.append(eval(f))
                final_result.append(instance_result)
            self.results_symbols = final_result

    def convert_model_responses_to_PyEuclidea_format(self, results):
        """
        Convert the generated responses to PyEuclidea Format.
        - results: A list of generated responses
        - results_tools: A list of only tools used from responses (Baselines + DEBUG)
        - results_symbols: A list of only symbols produced from responses (Baselines + DEBUG)
        Returns:
        - results: A list containing generated responses.
        """
        print("Formatting results...")
        self.results = []
        self.results_tools = []
        self.results_symbols = []
        for problem_instance, initial_symbols in zip(results, self.dataset['initial_symbol']):
            instance_responses = []
            instance_tools = []
            instance_symbols = []
            for generated_solution in problem_instance:
                # Some responses can be trash so filter them #
                if len(generated_solution) < 5:
                    continue
                response_solution, (response_tools, response_symbols) = decompose_example(solution=generated_solution,
                                                                                          initial_symbols=eval(
                                                                                              initial_symbols))
                instance_responses.append(response_solution)
                instance_tools.append(response_tools)
                instance_symbols.append(response_symbols)
            self.results.append(instance_responses)
            self.results_tools.append(instance_tools)
            self.results_symbols.append(instance_symbols)
        return

    def convert_and_send_to_PyEngine(self):
        """
        Convert results / results_tools / results_symbols to Tool Directives
        """
        tool_function_map = {
            'Line Tool': 'line_tool',
            'Perpendicular Tool': 'perp_tool',
            'Parallel Tool': 'parallel_tool',
            'Circle Tool': 'circle_tool',
            'Compass Tool': 'compass_tool',
            'Segment Tool': 'segment_tool',
            'Ray Tool': 'ray_tool',
            'Perpendicular Bisector Tool': 'perp_bisector_tool',
            'Angle Bisector Tool': 'angle_bisector_tool',
            'Midpoint Tool': 'midpoint_tool',
            'Intersection Tool': 'intersection_tool'
        }
        examples = len(self.results)
        active_examples = 0
        pass_at_k_function_calls = []
        for i in range(examples):
            active_examples += 1
            for instance_tools, instance_symbols in zip(self.results_tools, self.results_symbols):
                instance_function_calls = []
                for tools, symbols in zip(instance_tools, instance_symbols):
                    function_calls = []
                    for tool, symbol in zip(tools, symbols):
                        function_name = tool_function_map.get(tool, None)
                        if function_name:
                            function_calls.append([f"{function_name}", symbol])
                        else:
                            continue
                    instance_function_calls.append(function_calls)
                pass_at_k_function_calls.append(instance_function_calls)
        return pass_at_k_function_calls

    def match_tools(self, pack_list=None, threshold=1):
        raw_scores = []
        pack_results = {}
        # Find the n factor (n/k)
        examples = len(self.results_tools)
        active_examples = 0
        for i in range(examples):
            pack_as_string = match_index_with_pack(i, pack_list)
            if pack_list is None or pack_as_string not in pack_list:
                continue
            active_examples += 1
            attempts_per_example = len(self.results_tools[i])
            r = 0
            for j in range(attempts_per_example):
                scores = tool_overlap(response=self.results_tools[i][j],
                                      ground_truth=eval(self.dataset['solution_tool'].values[i]))
                raw_scores.append(scores)
                scores = 1.0 * (scores < threshold)
                r += scores
            pass1 = estimate_pass_at_k([attempts_per_example], [r], 1).mean()
            pass5 = estimate_pass_at_k([attempts_per_example], [r], 5).mean()
            pass20 = estimate_pass_at_k([attempts_per_example], [r], 20).mean()
            if pack_as_string not in pack_results:
                pack_results.update(
                    {pack_as_string: {'Exact Tool': {'pass@1': pass1, 'pass@5': pass5, 'pass@20': pass20}}})
            else:
                pack_results[pack_as_string]['Exact Tool']['pass@1'] += pass1
                pack_results[pack_as_string]['Exact Tool']['pass@5'] += pass5
                pack_results[pack_as_string]['Exact Tool']['pass@20'] += pass20

        ### PACK AVERAGE ###
        pass1_all = 0
        pass5_all = 0
        pass20_all = 0
        if pack_list is None:
            pack_list = [(k, eval(v)[1] - eval(v)[0] + 1) for v, k in self.pack_list.items()]
        else:
            pack_list = [(k, eval(v)[1] - eval(v)[0] + 1) for v, k in self.pack_list.items() if k in pack_list]
        for p, r in pack_list:
            if p not in pack_results:
                continue
            pass1_all += copy.deepcopy(pack_results[p]['Exact Tool']['pass@1'])
            pack_results[p]['Exact Tool']['pass@1'] = pack_results[p]['Exact Tool']['pass@1'] / r
            pass5_all += copy.deepcopy(pack_results[p]['Exact Tool']['pass@5'])
            pack_results[p]['Exact Tool']['pass@5'] = pack_results[p]['Exact Tool']['pass@5'] / r
            pass20_all += copy.deepcopy(pack_results[p]['Exact Tool']['pass@20'])
            pack_results[p]['Exact Tool']['pass@20'] = pack_results[p]['Exact Tool']['pass@20'] / r
        ### AVERAGE OVER ALL ###
        return {'Exact Tool': {'pass@1': pass1_all / active_examples, 'pass@5': pass5_all / active_examples,
                               'pass@20': pass20_all / active_examples}}, raw_scores

    def match_tools_and_symbols(self, pack_list=None, threshold=1, hardseq=False):
        raw_scores = []
        pack_results = {}
        # Find the n factor (n/k)
        examples = len(self.results_tools)
        active_examples = 0
        for i in range(examples):
            pack_as_string = match_index_with_pack(i, pack_list)
            if pack_list is None or pack_as_string not in pack_list:
                continue
            active_examples += 1
            attempts_per_example = len(self.results_tools[i])
            r = 0
            for j in range(attempts_per_example):
                scores = tool_overlap(response=self.results_tools[i][j],
                                      response_sym=self.results_symbols[i][j],
                                      ground_truth=eval(self.dataset['solution_tool'].values[i]),
                                      ground_truth_sym=eval(self.dataset['solution_symbol'].values[i]),
                                      hardseq=hardseq)
                raw_scores.append(scores)
                scores = 1.0 * (scores < threshold)
                r += scores
            pass1 = estimate_pass_at_k([attempts_per_example], [r], 1).mean()
            pass5 = estimate_pass_at_k([attempts_per_example], [r], 5).mean()
            pass20 = estimate_pass_at_k([attempts_per_example], [r], 20).mean()
            if pack_as_string not in pack_results:
                pack_results.update(
                    {pack_as_string: {
                        f'Exact ToolSym HardSeq:{hardseq}': {'pass@1': pass1, 'pass@5': pass5, 'pass@20': pass20}}})
            else:
                pack_results[pack_as_string][f'Exact ToolSym HardSeq:{hardseq}']['pass@1'] += pass1
                pack_results[pack_as_string][f'Exact ToolSym HardSeq:{hardseq}']['pass@5'] += pass5
                pack_results[pack_as_string][f'Exact ToolSym HardSeq:{hardseq}']['pass@20'] += pass20
        ### PACK AVERAGE ###
        pass1_all = 0
        pass5_all = 0
        pass20_all = 0
        if pack_list is None:
            pack_list = [(k, eval(v)[1] - eval(v)[0] + 1) for v, k in self.pack_list.items()]
        else:
            pack_list = [(k, eval(v)[1] - eval(v)[0] + 1) for v, k in self.pack_list.items() if k in pack_list]
        for p, r in pack_list:
            if p not in pack_results:
                continue
            pass1_all += copy.deepcopy(pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@1'])
            pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@1'] = \
                pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@1'] / r
            pass5_all += copy.deepcopy(pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@5'])
            pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@5'] = \
                pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@5'] / r
            pass20_all += copy.deepcopy(pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@20'])
            pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@20'] = \
                pack_results[p][f'Exact ToolSym HardSeq:{hardseq}']['pass@20'] / r
        ### AVERAGE OVER ALL ###
        return {f'Exact ToolSym HardSeq:{hardseq}': {'pass@1': pass1_all / active_examples,
                                                     'pass@5': pass5_all / active_examples,
                                                     'pass@20': pass20_all / active_examples}}, raw_scores

    def evaluate(self, results, pack_list, level_list):
        final_results = {}
        self.convert_model_responses_to_PyEuclidea_format(results=results)
        steps = self.convert_and_send_to_PyEngine()
        global_pass_at_k = 0.
        for step_sequence, pack, level in zip(steps, pack_list, level_list):
            gui_env = Drawing(load_level(level=level, level_pack=pack, win_size=(637, 490)))
            samples, correct = gui_env.run_llm_steps(step_sequence, pass_at_k=50)
            passrate = estimate_pass_at_k(samples, correct, len(steps[0]))
            global_pass_at_k += passrate
        final_results.update({f"pass@{len(steps[0])}": global_pass_at_k / len(level_list)})
        print(final_results)
        return

    def evaluate_lcs(self, pack_list=None):
        if self.results_tools is None:
            self.generate_random_baseline(random_name='lcs')
        final_results = {}
        final_results.update(self.match_tools(pack_list=pack_list)[0])
        print(final_results)
        return

    def evaluate_rollout(self, pack_list=None):
        if self.results_tools is None:
            self.generate_random_baseline(random_name='lcs')
        final_results = {}
        final_results.update(self.match_tools(pack_list=pack_list)[0])
        final_results.update(self.match_tools_and_symbols(pack_list=pack_list)[0])
        print(final_results)
        return

    def generate_random_baseline(self, random_name='rnd', number_of_generations=1):
        lookfor = f'{random_name}_{number_of_generations}.json'
        if os.path.exists(lookfor):
            with open(lookfor, 'r') as fin:
                results = json.load(fin)['results']
        else:
            if random_name == 'lcp':
                engine = LargestCommonPrefix()
                engine.fit(self.dataset)
                results = engine.predict_all(self.dataset)
                self.results_tools = []
                for problem_instance in results:
                    instance_tools = []
                    for response_tools in problem_instance:
                        instance_tools.append(response_tools)
                    self.results_tools.append(instance_tools)
            elif random_name == 'rollout':
                engine = RandomRollout()
                engine.fit(self.dataset)
                tools, symbols = engine.predict_all(self.dataset)
                self.results_tools = []
                self.results_symbols = []
                for problem_instance in tools:
                    instance = []
                    for response in problem_instance:
                        instance.append(response)
                    self.results_tools.append(instance)
                    self.results_symbols.append(instance)
            else:
                raise NotImplementedError
        return


if __name__ == "__main__":
    results_file = 'chatgpt35.json'
    with open(results_file, 'r') as fin:
        results = json.load(fin)['results']
    dataset_file = pd.read_csv('euclidea.csv').dropna().reset_index(drop=True).drop(columns=['Unnamed: 0'])
    evaluator = GeometryEvaluator(dataset=dataset_file)
    evaluator.evaluate(results, pack_list=['Alpha', 'Alpha'], level_list=['01_TEquilateral', '02_Angle60'])
