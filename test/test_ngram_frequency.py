import unittest
import pandas as pd
from ngram_frequency import *



class TestNGramFrequency(unittest.TestCase):
    def test_expand_texts(self):
        texts = ["you're a good boy",
                 'your boy is good']
        result = ["you are a good boy",
                 'your boy is good']
        expanded = expand_contractions(texts, w2v_dir='..\data')
        for i, e in enumerate(expanded):
            assert result[i] == e

        del expanded

    def test_preprocess_response(self):
        texts = ["you're a good boy. but i am not!!",
                 'if your boy is good, you will be blessed. Is it?']
        result = [['good', 'boy'],
                  ['boy', 'good', 'blessed']]
        processed = preprocess_responses(pd.Series(texts))
        for i, p in enumerate(processed):
            assert result[i] == p

    def test_get_ngrams_frequency(self):
        texts = ["you're a good boy. but i am not!!",
                 'if your boy is good, you will be blessed. Is it?']
        result = {'boy': 2, 'good': 2, 'blessed': 1}
        ngram_freqs = get_ngram_frequencies(pd.Series(texts))
        for key in ngram_freqs:
            assert result[key] == ngram_freqs[key]

    def test_get_ngrams(self):
        texts = ["you're a good boy. but i am not!!",
                 'if your boy is good, you will be blessed. Is it?']
        tokens = preprocess_responses(pd.Series(texts))
        print(get_ngrams(tokens[0], n=4))

    def test_get_freq_except(self):
        freqs = {'p': {'a': 1, 'b': 2, 'c': 3},
                 'q': {'a': 4, 'b': 5},
                 'r': {'a': 6, 'c': 7},
                 's': {'b': 8, 'c': 9, 'd': 10},
        }
        freq_rest = {'a': 10, 'b': 13, 'c': 16, 'd': 10}
        print(get_freq_except(freqs, 'p'))
        assert freq_rest == get_freq_except(freqs, 'p')

    def test_get_ngram_frequencies(self):
        data_dirs = ['09202021', '10082021', '10312021', '11092021']
        fnames = ['', '10082021', '10312021', '11092021']
        aggregate_response()
