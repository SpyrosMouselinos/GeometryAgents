from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def set_diff_sym(seta: set, setb: set):
    return len(seta.difference(setb)) + len(setb.difference(seta))


def list_flatten(somelist):
    new_list = []
    for f in somelist:
        if isinstance(f, list):
            for k in f:
                new_list.append(k)
        else:
            new_list.append(f)
    return new_list


def mayaralgo(matrix):
    # Convert the problem to a minimization problem
    cost_matrix = -1 * np.array(matrix)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Calculate the maximum score based on the optimal assignment
    max_score = -cost_matrix[row_indices, col_indices].sum()

    return max_score


def lcs_indices(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    i, j = len(a), len(b)
    res_a, res_b = [], []
    while i != 0 and j != 0:
        if lengths[i][j] == lengths[i - 1][j]:
            i -= 1
        elif lengths[i][j] == lengths[i][j - 1]:
            j -= 1
        else:
            res_a.append(i - 1)
            res_b.append(j - 1)
            i -= 1
            j -= 1
    ret_a = list(reversed(res_a))
    ret_b = list(reversed(res_b))
    assert len(ret_a) == len(ret_b)
    return ret_a, ret_b


def sanity_check_nl():
    response = ['Draw a line between the points.',
                'Using the compass draw a circle from point A.',
                'Draw another circle using point B as a center with the same radius.',
                'Find the point of intersection and connect them with a line.',
                'This line bisects the AB.'
                ]

    groundtruthA = ['Using the circle tool draw a circle with center A and radius equal to AB.',
                    'Using the circle tool draw a circle with center B and radius equal to AB.',
                    'Connect the points C and D using the line tool.',
                    'The line intersects AB to a point E that is the midpoint of AB'
                    ]

    groundtruthB = ['Using the circle tool draw a circle with center A and radius equal to AB.',
                    'Using the circle tool draw a circle with center B and radius equal to AB.',
                    'Intersecting the first circle at points C and D.',
                    'Use the line tool to connect points A and C.'
                    ]
    join = response + groundtruthA + groundtruthB
    paraphrases = util.paraphrase_mining(model, join,
                                         query_chunk_size=100,
                                         corpus_chunk_size=100000,
                                         max_pairs=len(join) ** 3,
                                         top_k=len(join))

    score_A = 0
    score_B = 0
    cache = {response[i]: [0, 0] for i in range(len(response))}
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        if join[i] in cache and (cache[join[i]][0] == 0 or cache[join[i]][1] == 0):
            if join[j] in groundtruthA and cache[join[i]][0] == 0:
                score_A += score
                cache[join[i]][0] += 1
            if join[j] in groundtruthB and cache[join[i]][1] == 0:
                score_B += score
                cache[join[i]][1] += 1

    print(f"Score A: {score_A / len(response)} / Score B: {score_B / len(response)}")
    join = [''.join(response)] + [''.join(groundtruthA)] + [''.join(groundtruthB)]
    paraphrases = util.paraphrase_mining(model, join,
                                         query_chunk_size=100,
                                         corpus_chunk_size=100000,
                                         max_pairs=3 ** 3,
                                         top_k=3)

    score_A_global = 0
    score_B_global = 0
    cache = {''.join(response): [0, 0]}
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        if join[i] in cache and (cache[join[i]][0] == 0 or cache[join[i]][1] == 0):
            if join[j] in [''.join(groundtruthA)] and cache[join[i]][0] == 0:
                score_A_global += score
                cache[join[i]][0] += 1
            if join[j] in [''.join(groundtruthB)] and cache[join[i]][1] == 0:
                score_B_global += score
                cache[join[i]][1] += 1

    print(f"Score A: {score_A * score_A_global / len(response)} / Score B: {score_B * score_B_global / len(response)}")


def best_matching_subset(response, ground_truth, all_cosine_scores):
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


def natual_language_overlap(response, ground_truth):
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
    sent_score, best_subset = best_matching_subset(response_split, ground_truth_split, emb_score)
    assert len(best_subset) == len(ground_truth_split)
    r = model.encode('\n'.join(best_subset), show_progress_bar=False, batch_size=1,
                     convert_to_tensor=True, device=DEVICE)
    g = model.encode(ground_truth, show_progress_bar=False, batch_size=1,
                     convert_to_tensor=True, device=DEVICE)
    sg = util.pytorch_cos_sim(r, g).cpu().numpy()[0][0]
    new_new_score = sent_score / len(best_subset) * sg
    return float(new_new_score)


def tool_overlap(response,
                 ground_truth,
                 disrupt_score=2,
                 extra_score=1,
                 missing_score=1,
                 symbol_mismatch_score=1,
                 response_sym=None,
                 ground_truth_sym=None,
                 hardseq=False):
    score = 0
    i, j = 0, 0

    ### Symbol Part ###
    if response_sym and ground_truth_sym:
        lcs_a, lcs_b = lcs_indices(response, ground_truth)

        if hardseq:
            while i < len(response) and j < len(ground_truth):
                if i in lcs_a and j in lcs_b:
                    if response_sym[i] != ground_truth_sym[j]:
                        score += symbol_mismatch_score
                    i += 1
                    j += 1
                elif response[i] == ground_truth[j]:
                    i += 1
                    j += 1
                else:
                    if response[i] in ground_truth[j:]:
                        score += disrupt_score
                        i += 1
                    else:
                        score += extra_score
                        i += 1
        else:
            ro = set(list_flatten([response_sym[k] for k in lcs_a]))
            go = set(list_flatten([ground_truth_sym[k] for k in lcs_b]))
            score += symbol_mismatch_score * set_diff_sym(ro, go)

    ### Tool Part ###
    i, j = 0, 0
    while i < len(response) and j < len(ground_truth):
        if response[i] == ground_truth[j]:
            i += 1
            j += 1
        else:
            if response[i] in ground_truth[j:]:
                score += disrupt_score
                i += 1
            else:
                score += extra_score
                i += 1

    while j < len(ground_truth):
        score += missing_score
        j += 1

    return score
