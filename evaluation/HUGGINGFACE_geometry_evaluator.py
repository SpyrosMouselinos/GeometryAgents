"""Euclidea Game Evaluation Metric."""
import re
import datasets
import evaluate
import Levenshtein
import copy
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import itertools
from itertools import combinations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

_DESCRIPTION = """
Natural Language Match Score: Given a geometric problem and a generated sequence of steps to solve it,
in natural language, this module computes a matching score between the ground truth and the provided solution.

The module works by segmenting the provided solution into steps. 
Then for each step it extracts the used tool and arguments to it. 
Finally it compares the solution steps with the ground truth ones using the Hungarian Matching Algorithm.
"""

TOOL2IDX = {
    'Perpendicular Tool': 0,
    'Line Tool': 1,
    'Circle Tool': 2,
    'Perpendicular Bisector Tool': 3,
    'Angle Bisector Tool': 4,
    'Parallel Tool': 5,
    'Compass Tool': 6,
    'Intersect Tool': 7,
    'Point Tool': 8,
}

IDX2TOOL = {
    v: k for k, v in TOOL2IDX.items()
}

DEFAULT_THRESHOLD = 0.63


class GeometryEvaluator:
    def __init__(self,
                 responses,
                 references):

        self.references = references
        if isinstance(responses, str):
            responses = [responses]
        elif isinstance(responses, list) and all([isinstance(f, list) for f in responses]):
            responses = responses[0]  # De-Nest it
        self.responses = responses

        if isinstance(references, str):
            references = [references]
        self.references = references

        # if isinstance(self.results_tools[0], str):
        #     self.results_tools = [[eval(f)] for f in self.results_tools]

        # if isinstance(self.results_symbols[0], str):
        #     self.results_symbols = [[eval(f)] for f in self.results_symbols]

    def get_symbols(self, text):
        # Define a regular expression to match single or double capital letter words
        pattern = r'\b[A-Z]{1,4}\b'

        # Find matches in the text using the regular expression and store their positions
        matches = [(match.group(), match.start()) for match in re.finditer(pattern, text)]

        # Sort the matches based on their positions in the text
        sorted_matches = sorted(matches, key=lambda x: x[1])

        # Extract the matched elements and return them in order
        elements = [match[0] for match in sorted_matches]

        return elements

    def compare_symbols(self, history, new):
        unique_elements = set(new) - set(history)
        return list(unique_elements)

    def check_word(self, word, sentence):
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, sentence, re.IGNORECASE))

    def annotate_solutions(self, step_solutions, toolset, initial_symbols=None):
        """
        Annotates solutions according to the tool used.
        Keeps track of emitted symbols at each tool round
        Heuristic method.
        """

        keyword_2_tool = {
            'circle': ['Circle Tool'],
            'line': ['Line Tool'],
            'point': ['Point Tool'],
            'intersect': ['Intersect Tool'],
            'bisect': ['Perpendicular Bisector Tool', 'Angle Bisector Tool'],
            'midpoint': ['Perpendicular Bisector Tool', 'Perpendicular Tool'],
            'angle': ['Angle Bisector Tool'],
            'vertical': ['Perpendicular Tool'],
            'perpendicular': ['Perpendicular Bisector Tool', 'Perpendicular Tool'],
            'parallel': ['Parallel Tool'],
            'compass': ['Compass Tool'],
        }

        step_solutions = [f for f in step_solutions.split('\n') if len(f) > 2]
        num_solutions = len(step_solutions)
        refined_solutions = []
        gt_tools = []
        history_symbols = initial_symbols if initial_symbols is not None else []
        per_step_symbols = [copy.deepcopy(history_symbols)]
        for i in range(num_solutions):
            current_solution = copy.deepcopy(step_solutions[i])
            current_solution = current_solution.lower().strip()
            if ',' in current_solution:
                current_solution = current_solution[:current_solution.find(',')]
            ### Voting classifier ###
            possible_tools = np.zeros(shape=(9,))  # 9 Tools
            for keyword, tools in keyword_2_tool.items():
                if self.check_word(keyword, current_solution):
                    for tool in tools:
                        if tool in toolset:
                            possible_tools[TOOL2IDX[tool]] += 1
            ### If no tool ###
            if np.sum(possible_tools) == 0:
                continue
            ### Take the smallest id as the most probable ###
            tool_name = IDX2TOOL[np.argmax(possible_tools)]
            refined_solutions.append(f'<{tool_name}>{step_solutions[i]}')
            gt_tools.append(tool_name)
            ### Now find (if any) associated points and resulting symbols from each operation ###
            emitted_symbols = self.get_symbols(step_solutions[i])
            new_symbols = self.compare_symbols(history_symbols, emitted_symbols)
            if len(new_symbols) > 0:
                history_symbols += new_symbols
            per_step_symbols.append(new_symbols)
        refined_solutions = '\n'.join(refined_solutions)
        return refined_solutions, gt_tools, per_step_symbols

    def format_solutions(self, solution):
        """
        qs: Part of the solution that usually is an assumption starting with Let.
        We move this to the question instead.
        """
        try:
            solution = solution.split('\n\n')[1].split('\n\n')[0]
        except:
            pass
        kw = None
        if 'Let' in solution:
            kw = 'Let'
        elif 'Given' in solution:
            kw = 'Given'
        if kw is not None:
            start_idx = solution.find(kw)
            ### Fix for one weird level ###
            offset = solution.find('{\displaystyle AB>AO,AC>AO}')
            if offset != -1:
                qs = 'Let O be the vertex of the angle and A the given point. Let B, C be abitary points on each ray, such that AB is bigger than AO and  AC is bigger than AO.'
                solution = 'Construct circle O with center O and radius OA.\nConstruct circle B with center B and radius BA, intersecting circle O at F.\nConstruct circle B with center C and radius CA, intersecting circle O at G.\nConstruct line FG, intersecting line OB at H, intersecting line OC at I.\nConstruct line AH.\nConstruct line AI.'
                return solution, qs
            else:
                possible_end_idxs = [solution.find(f) for f in ['.', 'Construct', 'Draw', 'With', 'Point', 'Starting']]
                possible_end_idxs = min([f for f in possible_end_idxs if f != -1]) + 1
            qs = solution[start_idx:possible_end_idxs]
            solution_ = solution[possible_end_idxs:]
            if solution_.startswith('onstruct'):
                possible_end_idxs -= 1
                qs = solution[start_idx:possible_end_idxs].strip()
            solution = solution[possible_end_idxs:]
        else:
            qs = None
        solution = solution.replace('\n\n', '\n')
        return solution, qs

    def format_tools(self, tools):
        proper_tools = []
        distances = np.zeros(shape=(9,))
        for tool in tools:
            if tool == 'Move Tool':
                continue
            ### Look over the correct tools ###
            for proper_tool_name, proper_tool_idx in TOOL2IDX.items():
                distances[proper_tool_idx] = Levenshtein.distance(tool.lower(), proper_tool_name.lower())
            ### Find the tool with the correct (minimum distance) ###
            proper_tools.append(IDX2TOOL[np.argmin(distances)])
        return proper_tools

    def decompose_example(self, solution, initial_symbols):
        s, _ = self.format_solutions(solution)
        tools = [k for k in TOOL2IDX.keys()]
        response_solution, response_tools, response_symbols = self.annotate_solutions(s, tools, initial_symbols)
        return response_solution, (response_tools, response_symbols)

    def evaluate(self):
        instance_responses = []
        instance_tools = []
        instance_symbols = []
        for generated_solution in self.responses:
            if len(generated_solution) < 10:
                continue
            response_solution, (response_tools, response_symbols) = self.decompose_example(
                solution=generated_solution,
                initial_symbols=None)
            instance_responses.append(response_solution)
            instance_tools.append(response_tools)
            instance_symbols.append(response_symbols)
        return instance_responses, instance_tools, instance_symbols

    def best_matching_subset(self, response, ground_truth, all_cosine_scores):
        max_score = float('-inf')
        best_subset = None

        # Iterate over all subsets of response of the required size
        for subset_indices in combinations(range(len(response)), len(ground_truth)):
            # Select the scores for the current subset
            subset_scores = all_cosine_scores[np.ix_(subset_indices, range(len(ground_truth)))]

            # Convert to negative for the cost matrix
            cost_matrix = -subset_scores

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_score = -cost_matrix[row_ind, col_ind].sum()

            # Update max_score and best_subset if this is the best so far
            if total_score > max_score:
                max_score = total_score
                best_subset = [response[i] for i in subset_indices]
        if best_subset is None:
            return 0, [''] * len(ground_truth)
        return max_score, best_subset

    def estimate_pass_at_k(self, num_samples, num_correct, k, ):
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

    def nl_overlap(self, response, ground_truth):
        response_split = response.split('\n')
        ground_truth_split = ground_truth.split('\n')
        if len(response_split) < len(ground_truth_split):
            return 0
        ###################################################################################
        response_emb = model.encode(response_split, show_progress_bar=False, batch_size=len(response_split),
                                    convert_to_tensor=True, device=DEVICE)
        gt_emb = model.encode(ground_truth_split, show_progress_bar=False, batch_size=len(ground_truth_split),
                              convert_to_tensor=True, device=DEVICE)
        emb_score = util.pytorch_cos_sim(response_emb, gt_emb).cpu().numpy()
        sent_score, best_subset = self.best_matching_subset(response_split, ground_truth_split, emb_score)
        assert len(best_subset) == len(ground_truth_split)
        r = model.encode('\n'.join(best_subset), show_progress_bar=False, batch_size=1,
                         convert_to_tensor=True, device=DEVICE)
        g = model.encode(ground_truth, show_progress_bar=False, batch_size=1,
                         convert_to_tensor=True, device=DEVICE)
        sg = util.pytorch_cos_sim(r, g).cpu().numpy()[0][0]
        new_new_score = sent_score / len(best_subset) * sg
        return float(new_new_score)

    def test_1(self, instance_responses):
        references = self.references
        raw_scores = []
        for j in range(len(instance_responses)):
            scores = self.nl_overlap(instance_responses[j], references[0])
            raw_scores.append(scores)
        return raw_scores

    def calc_thresh_pass(self, raw_scores, threshold=0.65):
        r = 0
        for score in raw_scores:
            r += 1.0 * (score > threshold)
        pass1 = self.estimate_pass_at_k([len(raw_scores)], [r], 1).mean()
        pass10 = self.estimate_pass_at_k([len(raw_scores)], [r], 10).mean()
        pass25 = self.estimate_pass_at_k([len(raw_scores)], [r], 25).mean()
        pass50 = self.estimate_pass_at_k([len(raw_scores)], [r], 50).mean()
        return pass1, pass10, pass25, pass50


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class EuclideaGameEval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Value("string"),
                }
            )
        )

    def _compute(self, predictions, references):
        ge = GeometryEvaluator(responses=predictions, references=references)
        tmp_, _, _ = ge.evaluate()
        valid_k = len(tmp_)
        pass1, pass10, pass25, pass50 = ge.calc_thresh_pass(ge.test_1(tmp_), DEFAULT_THRESHOLD)
        return {
            "pass@1": pass1,
            "pass@10": pass10 if valid_k >= 10 else None,
            "pass@25": pass25 if valid_k >= 25 else None,
            "pass@50": pass50 if valid_k >= 50 else None
        }
