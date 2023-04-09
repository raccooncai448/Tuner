import numpy as np
from functools import reduce
import string

def process(text):
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.translate(str.maketrans('','','1234567890'))
    text = text.lower()
    return text


def get_data(train_file, test_file):
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    with open(train_file) as temp:
        train = temp.read()
    with open(test_file) as temp:
        test = temp.read()

    sentence_list = (train).split() + (test).split()
    unique_words = sorted(set(sentence_list))

    w2t_dict = {w:i for i, w in enumerate(unique_words)}

    train_data, test_data = [w2t_dict[w] for w in train.split()], [w2t_dict[w] for w in test.split()]
    vocabulary = w2t_dict

    return train_data, test_data, vocabulary
