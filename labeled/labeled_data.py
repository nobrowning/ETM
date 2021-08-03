import os
import random
import pickle
import numpy as np
import torch 
import scipy.io


def _fetch(path, name):
    token_file = os.path.join(path, '{}_tokens.mat'.format(name))
    count_file = os.path.join(path, '{}_counts.mat'.format(name))
    tag_file = os.path.join(path, '{}_tags.mat'.format(name))

    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    tags = scipy.io.loadmat(tag_file)['tags'].squeeze()

    return {'tokens': tokens, 'counts': counts, 'tags': tags}


def get_data(path):
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    train = _fetch(path, 'train')
    valid = _fetch(path, 'valid')
    test = _fetch(path, 'test')
    test_h1 = _fetch(path, 'test_h1')
    test_h2 = _fetch(path, 'test_h2')

    test.update({'tokens_1': test_h1['tokens'], 'counts_1': test_h1['counts'], 'tags_1': test_h1['tags'],
                 'tokens_2': test_h2['tokens'], 'counts_2': test_h2['counts'], 'tags_2': test_h2['tags']})

    return vocab, train, valid, test


def get_batch(tokens, counts, tags, ind, vocab_size, device, emsize=300):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    tag_batch = []
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        tag = tags[doc_id]
        L = count.shape[1]
        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
            tag = [tag.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
            tag = tag.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
            tag_batch.append(tag)
    try:
        tag_batch = np.vstack(tag_batch)
    except:
        print(tag_batch)
    data_batch = torch.from_numpy(data_batch).float().to(device)
    tag_batch = torch.from_numpy(tag_batch).float().to(device)
    return data_batch, tag_batch
