import random
from collections import Counter
from itertools import islice, tee


class LargestCommonPrefix:
    def __init__(self):
        self.idx2tool = {0: 'Perpendicular Bisector Tool',
                         1: 'Line Tool',
                         2: 'Parallel Tool',
                         3: 'Point Tool',
                         4: 'Perpendicular Tool',
                         5: 'Angle Bisector Tool',
                         6: 'Circle Tool',
                         7: 'Compass Tool'}
        self.tool2idx = {'Perpendicular Bisector Tool': 0,
                         'Line Tool': 1,
                         'Parallel Tool': 2,
                         'Point Tool': 3,
                         'Perpendicular Tool': 4,
                         'Angle Bisector Tool': 5,
                         'Circle Tool': 6,
                         'Compass Tool': 7}

    def find_top_common_sequences(self, lists):
        def longest_common_subsequence(X, Y):
            """Function to find the longest common subsequence in two lists of integers."""
            m = len(X)
            n = len(Y)
            L = [[0] * (n + 1) for i in range(m + 1)]

            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])

            # Following code is used to print LCS
            index = L[m][n]
            lcs = [0] * (index + 1)
            lcs[index] = ""

            # Start from the right-most-bottom-most corner and
            # one by one store characters in lcs[]
            i = m
            j = n
            while i > 0 and j > 0:
                if X[i - 1] == Y[j - 1]:
                    lcs[index - 1] = X[i - 1]
                    i -= 1
                    j -= 1
                    index -= 1
                elif L[i - 1][j] > L[i][j - 1]:
                    i -= 1
                else:
                    j -= 1

            return lcs[:-1]  # Exclude the last empty string

        # Find all common subsequences for each pair of lists
        common_sequences = []
        for i in range(len(lists)):
            for j in range(i + 1, len(lists)):
                lcs = longest_common_subsequence(lists[i], lists[j])
                if lcs:
                    common_sequences.append(lcs)

        # Sort the common sequences by length and return the top 5
        common_sequences.sort(key=len, reverse=True)
        top_5 = common_sequences[:5]
        named_top_5 = []
        for line in top_5:
            named_top_5.append([self.idx2tool[f] for f in line])
        return named_top_5

    def fit(self, data):
        x = data.dropna()['solution_tool'].values
        y = [eval(f) for f in x]
        self.enumerated_y = []
        for line in y:
            enumerated_line = [self.tool2idx[f] for f in line]
            self.enumerated_y.append(enumerated_line)

        self.named_top_5 = self.find_top_common_sequences(self.enumerated_y)

    def predict(self):
        if hasattr(self, 'named_top_5'):
            return random.choice(self.named_top_5)
        else:
            print("You need to use .fit() first!\n")

    def predict_all(self, data):
        out = len(data)
        results = []
        for i in range(out):
            results.append([self.predict() for _ in range(50)])
        return results


class RandomRollout:
    def __init__(self):
        self.idx2tool = {0: 'Perpendicular Bisector Tool',
                         1: 'Line Tool',
                         2: 'Parallel Tool',
                         3: 'Point Tool',
                         4: 'Perpendicular Tool',
                         5: 'Angle Bisector Tool',
                         6: 'Circle Tool',
                         7: 'Compass Tool',
                         8: 'Intersect Tool'}
        self.tool2idx = {'Perpendicular Bisector Tool': 0,
                         'Line Tool': 1,
                         'Parallel Tool': 2,
                         'Point Tool': 3,
                         'Perpendicular Tool': 4,
                         'Angle Bisector Tool': 5,
                         'Circle Tool': 6,
                         'Compass Tool': 7,
                         'Intersect Tool': 8
                         }
        self.sym2idx = {'BD_O': 0, 'AB_CD': 1, 'ACB_D': 2, 'AB_A': 3, 'BAC_C': 4, 'EF_G_AE': 5, 'BG': 6, 'BE_H_AG': 7,
                        'BE_ABC': 8, 'AC_E_D': 9, 'A_B': 10, 'O': 11, 'A_O_A_B_C_AB_AO_AC_AO': 12, 'O_A_B_C': 13,
                        'C_OA': 14, 'EA_O': 15, 'A_B_C': 16, 'BO_D': 17, 'AB_BM': 18, 'C_B_D': 19, 'D_AD': 20, 'F': 21,
                        'C_BC_D': 22, 'O_A': 23, 'E_D_AB': 24, 'AB_O_O_O_AO_BO': 25, 'B_C_O': 26, 'C_D': 27,
                        'AF_G_BD': 28, 'AB_E': 29, 'F_DE': 30, 'G_AF': 31, 'H_BD_AG': 32, 'F_AB_E': 33, 'F_EB_E': 34,
                        'EH': 35, 'BO_B_C': 36, 'AG': 37, 'BC_E_D': 38, 'BD_E': 39, 'D_AC': 40, 'D_E_AC_B': 41,
                        'F_AD_AE': 42, 'AF': 43, 'E_AC': 44, 'BD': 45, 'OD': 46, 'CO_E_D': 47, 'OF_G': 48,
                        'O_A_B_O': 49, 'CA_E_C': 50, 'BO_C': 51, 'OB_OC_FG_H_I': 52, 'CB': 53, 'HI': 54, 'H_CG_I': 55,
                        'F_BD_E': 56, 'G_CA': 57, 'G': 58, 'F_G_CE': 59, 'BE_F': 60, 'E_AB': 61, 'BC_D': 62, 'H': 63,
                        'E_CD': 64, 'D_AB_C_B': 65, 'AB_C': 66, 'C': 67, 'OE': 68, 'F_B': 69, 'CAD': 70, 'OB_OA': 71,
                        'AI': 72, 'FH_J': 73, 'AB_D': 74, 'BAC': 75, 'EG': 76, 'B_C': 77, 'E_D_ABD': 78, 'O_O_A': 79,
                        'AC': 80, 'H_G_BG': 81, 'DAC': 82, 'ABC_M_D_BA_E_BC_DM_ME_BD_DM_ME': 83, 'H_G_B': 84,
                        'ABCD_AB': 85, 'ABCD_AB_AD': 86, 'BF': 87, 'DAE': 88, 'F_AE': 89, 'EF_G_CD': 90, 'AD_E_CD': 91,
                        'D_BA': 92, 'AE': 93, 'AB_C_B': 94, 'ABCD_EFGH': 95, 'G_AE': 96, 'G_DF': 97, 'EF': 98,
                        'G_BC': 99, 'BC_E_A': 100, 'I_CD': 101, 'AC_B': 102, 'DB_D': 103, 'CE': 104, 'AB_D_E': 105,
                        'C_AB_AC_AB': 106, 'F_BC': 107, 'AO': 108, 'F_AE_E': 109, 'F_G_AC_ABE': 110, 'BO': 111,
                        'D_OC_OD_O_A_B_C': 112, 'F_CD': 113, 'E_D_AB_C_BC_A': 114, 'AB_C_AB': 115, 'E_D_B': 116,
                        'B_A': 117, 'B_C_A': 118, 'G_BF_CD': 119, 'DG': 120, 'IJ': 121, 'B_OA': 122, 'AD_E': 123,
                        'AE_A': 124, 'B': 125, 'EJ_K': 126, 'CAB': 127, 'CD': 128, 'A_B_C_A_B': 129, 'BE': 130,
                        'B_C_OA': 131, 'OB_C': 132, 'DE': 133, 'CF': 134, 'AD_D': 135, 'C_AC_BC': 136,
                        'A_B_B_A_B_A': 137, 'F_EBA_E': 138, 'F_BF': 139, 'DAB_AB_D': 140, 'EF_G': 141, 'C_OB_E_D': 142,
                        'O_ABCD_A': 143, 'C_A': 144, 'BD_I': 145, 'AC_BC_D': 146, 'E': 147, 'F_BA': 148, 'D': 149,
                        'AB_CD_EF': 150, 'FG': 151, 'AC_AB_C_B': 152, 'ABCD': 153, 'CAB_C': 154, 'B_D': 155, 'ABC': 156,
                        'F_E_AC_CD': 157, 'OA': 158, 'G_CE': 159, 'C_D_OA': 160, 'OC': 161, 'AB': 162, 'C_AC': 163,
                        'AC_D': 164, 'FH_I_J': 165, 'F_E_CD': 166, 'C_AB_D_E': 167, 'AD': 168, 'C_A_B_A_B_C_C_A_B': 169,
                        'F_EC': 170, 'DF': 171, 'AOB': 172, 'BAC_D': 173, 'A_ABCD_E': 174, 'ABCD_O': 175, 'AH': 176,
                        'BC': 177, 'AD_E_D': 178, 'I_BG': 179, 'F_DE_AC': 180, 'AG_EG': 181, 'AB_O': 182,
                        'AB_C_AB_AC': 183, 'ABCD_A': 184, 'CF_G': 185, 'O_A_B': 186, 'A': 187, 'AD_A': 188, 'AO_A': 189,
                        'F_AB': 190, 'C_AB_D': 191, 'F_E': 192}
        self.idx2sym = {v: k for k, v in self.sym2idx.items()}

    def fix_symbols(self, y_clean):
        new_y = []
        for line in y_clean:
            new_line = []
            for item in line:
                if len(item) > 1:
                    new_item = '_'.join(item)
                else:
                    try:
                        new_item = item[0]
                    except:
                        pass
                new_line.append(new_item)
            new_y.append(new_line)
        return new_y

    def backtranslate_symbols(self, y_clean):
        new_y = []
        for line in y_clean:
            new_line = []
            for item in line:
                if '_' in item:
                    new_item = [f for f in item.split('_')]
                else:
                    new_item = [item]
                new_line.append(new_item)
            new_y.append(new_line)
        return new_y

    def fit(self, data):
        x = data.dropna()['question'].apply(lambda x: x.split('Tool List: ')[-1].split('Solution:')[0].strip())
        xz = data.dropna()['solution_symbol']
        y = [eval(f) for f in x.values]
        z = self.fix_symbols([eval(f) for f in xz.values])
        self.enumerated_y = []
        self.enumerated_z = []
        for line_y, line_z in zip(y, z):
            enumerated_liney = [self.tool2idx[f] for f in line_y]
            enumerated_linez = [self.sym2idx[f] for f in line_z]
            self.enumerated_y.append(enumerated_liney)
            self.enumerated_z.append(enumerated_linez)
        self.ngram_tool_frequency_tables(self.enumerated_y)
        self.ngram_symbol_frequency_tables(self.enumerated_z)
        return

    def ngram_tool_frequency_tables(self, list_of_lists):
        # Flatten the list of lists
        flat_list = [item for sublist in list_of_lists for item in sublist]

        # Generate unigrams, bigrams, and trigrams
        self.unigrams = Counter(flat_list)
        self.bigrams = Counter(zip(flat_list, islice(flat_list, 1, None)))
        self.trigrams = Counter(zip(*[islice(seq, i, None) for i, seq in enumerate(tee(flat_list, 3))]))

    def ngram_symbol_frequency_tables(self, list_of_lists):
        # Flatten the list of lists
        flat_list = [item for sublist in list_of_lists for item in sublist]

        # Generate unigrams, bigrams, and trigrams
        self.unigramss = Counter(flat_list)
        self.bigramss = Counter(zip(flat_list, islice(flat_list, 1, None)))
        self.trigramss = Counter(zip(*[islice(seq, i, None) for i, seq in enumerate(tee(flat_list, 3))]))

    def sample_from_tool_table(self, n, k):
        if n == 1:
            elements, weights = zip(*self.unigrams.items())
        elif n == 2:
            # Sampling from bigrams
            elements, weights = zip(*self.bigrams.items())
        elif n == 3:
            # Sampling from trigrams
            elements, weights = zip(*self.trigrams.items())
        else:
            raise ValueError("n must be 1, 2, or 3")

        # Draw k samples based on the frequency weights
        return random.choices(elements, weights, k=k)

    def sample_from_sym_table(self, n, k):
        if n == 1:
            elements, weights = zip(*self.unigramss.items())
        elif n == 2:
            # Sampling from bigrams
            elements, weights = zip(*self.bigramss.items())
        elif n == 3:
            # Sampling from trigrams
            elements, weights = zip(*self.trigramss.items())
        else:
            raise ValueError("n must be 1, 2, or 3")

        # Draw k samples based on the frequency weights
        return random.choices(elements, weights, k=k)

    def predict(self, idx, n=1, k=100):
        if idx < 0 or idx >= len(self.enumerated_y):
            print(f"Choose a number between 0 and {len(self.enumerated_y)}")
        # available_tools = self.enumerated_y[idx]
        ### Pick Iteratively with proportionality of uni/bi/trigrams ###
        tools = []
        symbols = []
        for _ in range(k):
            g = self.sample_from_tool_table(n, 7)
            z = self.sample_from_sym_table(n, 7)
            if n > 1:
                g = [l for f in g for l in f]
                z = [l for f in z for l in f]
            tools.append([self.idx2tool[i] for i in g][:7])
            symbols.append([self.idx2sym[i] for i in z][:7])
        symbols = self.backtranslate_symbols(symbols)
        return tools, symbols

    def predict_all(self, data, n=3, k=100):
        out = len(data)
        tools = []
        symbols = []
        for i in range(out):
            t, s = self.predict(idx=i, n=n, k=k)
            tools.append(t)
            symbols.append(s)
        return tools, symbols
