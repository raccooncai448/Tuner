import os
import torch
import datasets
import transformers
import json

from datasets import load_dataset
from preprocess import get_data
from scraper import generate
from finetune import finetune

PATH = 'cache'
LOG_FILE = 'cache/corpus.txt'
DATA_FILE = 'cache/data.json'
BREADTH = 4

def data_generator():
    try:
        os.mkdir(PATH)
    except:
        pass
    dict = generate(BREADTH, LOG_FILE)
    _, _, vocab = get_data(LOG_FILE, LOG_FILE)

    tok = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    model = transformers.AutoModelForCausalLM.from_pretrained('gpt2')

    #for count, word in enumerate(vocab.keys()):
        #print(word)
        #print(tok.convert_ids_to_tokens(tok(word)['input_ids']))

    oov_list = []
    for word in vocab.keys():
        if (word != tok.convert_ids_to_tokens(tok(word)['input_ids'])):
            oov_list.insert(0, word)
    assert len(oov_list) > 0
    print(f'Found {len(oov_list)} untokenized words!')