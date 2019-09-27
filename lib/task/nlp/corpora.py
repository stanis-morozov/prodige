"""
Datasets are plaintext collections. The only thing required for dataset is to iterate over raw strings of text.
Good source of datasets: https://github.com/niderhoff/nlp-datasets
"""
import os
import os.path as osp

import gensim
import nltk

from ...utils.download import download

ROOT_PATH = osp.join(*osp.abspath(__file__).split(osp.sep)[:-2])
DATA_PATH = osp.join(ROOT_PATH, 'data')
os.system('mkdir -p ' + DATA_PATH)


class FileDataset:
    def __init__(self, path, tokenizer=nltk.RegexpTokenizer(r"\b[a-zA-Z]{2,}\b")):
        self.path = path
        self.tokenizer = tokenizer

    def __iter__(self):
        lines = list(open(self.path))
        lines_tok = self.tokenizer.tokenize_sents(map(str.lower, lines))
        return iter(lines_tok)


class Wikipedia:
    def __init__(
            self, path=osp.join(DATA_PATH, 'enwiki-latest-pages-articles.xml.bz2'),
            url='https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2',
    ):
        self.path, self.url = path, url
        if not osp.exists(self.path):
            download(self.url, self.path)

        self.corpus = gensim.corpora.WikiCorpus(self.path)

    def __iter__(self):
        for text in self.corpus.get_texts():
            yield text
