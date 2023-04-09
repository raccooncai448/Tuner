import torch
import datasets
import transformers
import json
import os

from datasets import load_dataset
from preprocess import get_data, process
from scraper import generate
from finetune import finetune

try:
    os.mkdir('cache')
except:
    pass

LOG_FILE = 'cache/corpus.txt'
DATA_FILE = 'cache/data.json'
REF_FILE = 'cache/ref.json'
BREADTH = 200

def main():
    ### TO-DO: Generate data and vocab from wikipedia articles
    dict, ref = generate(BREADTH, LOG_FILE)
    _, _, vocab = get_data(LOG_FILE, LOG_FILE)

    ### TO-DO: Sift out new vocab and tokenize them
    tok = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    model = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
    
    #for count, word in enumerate(vocab.keys()):
    #    print(word)
    #    print(tok.convert_ids_to_tokens(tok(word)['input_ids']))
    
    
    oov_list = []
    with open('cache/new_words.txt', "w") as outfile:
        for word in vocab.keys():
            if len(tok.convert_ids_to_tokens(tok(word)['input_ids'])) > 1 or (process(word) != process(tok.convert_ids_to_tokens(tok(word)['input_ids'])[0])):
                    outfile.write(word + ' --- ')
                    outfile.write(tok.convert_ids_to_tokens(tok(word)['input_ids'])[0])
                    outfile.write('\n')
                    oov_list.insert(0, word)
    oov_list = list(set(oov_list))
    assert len(oov_list) > 0
    print(f'Found {len(oov_list)} untokenized words! Saved to new_words.txt')


    '''tok.add_tokens(oov_list)
    model.resize_token_embeddings(len(tok))

    ### TO-DO: Avg. Embedding and Dataset Initialization
    params = model.state_dict()
    embeddings = params['transformer.wte.weight']
    pre_expansion_embeddings = embeddings[:-3,:]
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5*sigma)

    new_embeddings = torch.stack(tuple((dist.sample() for _ in range(len(oov_list)))), dim=0)
    embeddings[-len(oov_list):,:] = new_embeddings
    params['transformer.wte.weight'][-len(oov_list):,:] = new_embeddings
    model.load_state_dict(params)'''

    ### TO-DO: Fine-tune model using custom dataset and embedding initializations
    jsonified = json.dumps(dict, indent=4)
    with open(DATA_FILE, "w") as outfile:
        outfile.write(jsonified)

    jsonified = json.dumps(ref, indent=4)
    with open(REF_FILE, "w") as outfile:
        outfile.write(jsonified)

    load_dataset("json", data_files=DATA_FILE)
    finetune(oov_list)

if __name__ == '__main__':
    main()