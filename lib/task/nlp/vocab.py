from collections import Counter
import numpy as np


class Vocab:
    def __init__(self, tokens, pad="_PAD_", unk='_UNK_'):
        """
        A special class that converts lines of tokens into matrices and backwards
        """
        assert all(tok in tokens for tok in (pad, unk))
        self.tokens = tokens
        self.token_to_ix = {t:i for i, t in enumerate(tokens)}
        self.pad, self.unk = pad, unk
        self.pad_ix = self.token_to_ix.get(pad)
        self.unk_ix = self.token_to_ix.get(unk)

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_lines(lines, pad="_PAD_", unk='_UNK_', max_tokens=None, min_count=0):
        flat_lines = '\n'.join(list(lines)).split()
        token_counts = Counter(list(flat_lines))
        tokens = sorted(token for token, count in token_counts.most_common(max_tokens) if count >= min_count)
        tokens = [t for t in tokens if t not in (pad, unk) and len(t)]
        tokens = [pad, unk] + tokens
        return Vocab(tokens, pad, unk)

    def tokenize(self, string):
        """converts string to a list of tokens"""
        tokens = [tok if tok in self.token_to_ix else self.unk
                  for tok in string.split()]
        return tokens

    def to_matrix(self, lines, max_len=None):
        """
        convert variable length token sequences into  fixed size matrix
        example usage:
        >>>vocab, lines = Vocab(...), list(...)
        >>>print(vocab.to_matrix(lines))
        [[15 22 21 28 27 13 -1 -1 -1 -1 -1]
         [30 21 15 15 21 14 28 27 13 -1 -1]
         [25 37 31 34 21 20 37 21 28 19 13]]
        """
        lines = list(map(self.tokenize, lines))
        max_len = max_len or max(map(len, lines))
        matrix = np.full((len(lines), max_len), self.pad_ix, dtype='int32')
        for i, seq in enumerate(lines):
            row_ix = list(map(self.token_to_ix.get, seq))[:max_len]
            matrix[i, :len(row_ix)] = row_ix
        return matrix

    def to_lines(self, matrix, crop=True):
        """
        Convert matrix of token ids into strings
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        :param crop: if True, crops PAD from line
        :return:
        """
        lines = []
        for line_ix in map(list,matrix):
            if crop:
                if self.pad_ix in line_ix:
                    line_ix = line_ix[:line_ix.index(self.pad_ix)]
            line = ' '.join(self.tokens[i] for i in line_ix)
            lines.append(line)
        return lines
