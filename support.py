import re
import string
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer


def process_tweet(tweet):
    """
    Preproceeses a Tweet by removing hashes, RTs, @mentions,
    links, stopwords and punctuation, tokenizing and stemming 
    the words.

    Accepts:
        tweet {str} -- tweet string

    Returns:
        {list<str>}
    """

    proc_twt = re.sub(r'^RT[\s]+', '', tweet)
    proc_twt = re.sub(r'@[\w_-]+', '', proc_twt)
    proc_twt = re.sub(r'#', '', proc_twt)
    proc_twt = re.sub(r'https?:\/\/.*[\r\n]*', '', proc_twt)
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True
    )

    twt_clean = []
    twt_tokens = tokenizer.tokenize(proc_twt)
    stopwords_en = stopwords.words('english')
    for word in twt_tokens:
        if word not in stopwords_en and word not in string.punctuation:
            twt_clean.append(word)

    twt_stems = []
    stemmer = PorterStemmer()
    for word in twt_clean:
        twt_stems.append(stemmer.stem(word))

    return twt_stems


def extract_features(tweet, freqs):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1, 2)
    '''
    word_l = process_tweet(tweet)
    x = np.zeros((1, 2))
    for word in word_l:
        x[0, 0] += freqs.get((word, 1), 0)
        x[0, 1] += freqs.get((word, 0), 0)
    assert(x.shape == (1, 2))
    return x


freqs = joblib.load('freqs.pkl')
model = joblib.load('model.pkl')
