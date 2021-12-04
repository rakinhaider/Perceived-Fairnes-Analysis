import string

import pandas

from survey_response_aggregator import aggregate_response
from combine import get_parser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from pycontractions import Contractions
import contractions
import os
from collections import Counter
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer


def get_contractor(data_dir, w2v='t'):
    if w2v == 'g':
        w2vec_path = os.path.join(
            data_dir, 'GoogleNews-vectors-negative300.bin')
        cont = Contractions(w2vec_path)
    else:
        cont = Contractions(api_key="glove-twitter-25")
    print('model loading', flush=True)
    cont.load_models()
    print('model loading finished', flush=True)
    return cont


def expand_contractions(texts, w2v_dir='data', method='naive'):
    if method == 'naive':
        return [contractions.fix(t) for t in texts if not pandas.isna(t)]
    else:
        cont = get_contractor(w2v_dir)
        expanded_text = cont.expand_texts(texts, precise=False)
        return expanded_text


def get_tokens(text):
    """
        1. Tokenize in to words.
        2. Remove punctuations.
        3. Remove stopwords.

    :param text:
    :return:
    """
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    stop_words.update(['would'])
    tokens = [w for w in words if not w in stop_words]
    return tokens


def preprocess_responses(response):
    """
        Take a Series of textual responses and perform the following.
        1. Remove Contractions.
        2. Split into sentences.
        3. Tokenize into words.
        4. Perform stemming.
    :param response: pandas.Series of participant responses.
    :return: list of sentences.
    """
    texts = response.values
    texts = [t for t in expand_contractions(texts)]
    sentences = []
    for t in texts:
        sentences.extend(sent_tokenize(t))
    tokens = [get_tokens(s) for s in sentences]
    tokens = [t for t in tokens if len(t) > 0]
    wl = WordNetLemmatizer()
    tokens = [[wl.lemmatize(w, 'n') for w in t] for t in tokens]
    return tokens


def get_ngrams(tokens, n):
    n_grams = []
    for i in range(n, max(len(tokens), n) + 1):
        n_grams.append(' '.join(tokens[i-n:i]))
    return n_grams


def get_ngram_frequencies(responses, n=1):
    """
        This method takes a pandas.Series of textual responses.
        Then performs the following.
        1. Process responses. Tokenize words.
        2. Perform stemming (Optional)
        3. Count n-gram frequency.

    :param responses:
    :param n:
    :return:
    """
    tokens = preprocess_responses(responses)
    counter = Counter()
    for t in tokens:
        ngrams = get_ngrams(t, n)
        print(ngrams)
        counter.update(ngrams)
    return counter


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    fnames = args.fnames
    fnames = [f + '_approved.csv' for f in fnames]

    response = aggregate_response(args.resp_dirs, fnames)
    for tup, grp in response.groupby(args.criteria):
        print(tup)
        freq = get_ngram_frequencies(grp[args.qid], n=2)
        print(freq)
        freq = {key: freq[key] for key in freq if freq[key] >= 2}
        wcld = WordCloud(background_color='white',
                         width=800, height=400)
        wcld.generate_from_frequencies(frequencies=freq)
        file_dir = os.path.join('outputs', '_'.join(args.resp_dirs),
                                'wordclouds')
        os.makedirs(file_dir, exist_ok=True)
        wcld.to_file(os.path.join(file_dir, '{}_{}.pdf'.format(tup, args.qid)))