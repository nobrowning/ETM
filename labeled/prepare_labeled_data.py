import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from scipy import sparse
from scipy.io import savemat
import pickle
from pathlib import Path
import os
from typing import Dict, List, Set, Tuple, Any
import logging
from tqdm.auto import tqdm


class LabeledDataConvertor:
    def __init__(self, sentences_with_tags: List[Tuple[str, List[str]]],
                 test_size_rate=0.01, valid_size_rate=0.01):
        self.cvectorizer = \
            CountVectorizer(min_df=10, max_df=0.9, token_pattern=r'[A-Za-z0-9\-_]+', stop_words=None)
        self.word2id: Dict[str, int]
        self.id2word: Dict[int, str]
        self.test_size_rate, self.valid_size_rate = test_size_rate, valid_size_rate
        # self.sentences_with_tags: List[Tuple[str, List[str]]] = sentences_with_tags
        self.sentences = []
        self.tags_list = []
        for s, t in sentences_with_tags:
            self.sentences.append(s)
            self.tags_list.append(t)
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([["machine_learning", "natural_language_processing",
                       "information_retrieval", "database", "data_mining",
                       "world_wide_web", "computer_vision"]])
        self.vocab: List[str]

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                # logging.FileHandler("bert_cls.log"),
                                logging.StreamHandler()
                            ],
                            level=logging.INFO)

    def build_vocabulary_and_label_transfer(self):
        cvz = self.cvectorizer.fit_transform(self.sentences).sign()
        logging.info('num of docs: {}'.format(cvz.shape[0]))
        logging.info('num of words: {}'.format(cvz.shape[1]))

        sum_counts = cvz.sum(axis=0)
        v_size = sum_counts.shape[1]
        sum_counts_np = np.zeros(v_size, dtype=int)
        for v in range(v_size):
            sum_counts_np[v] = sum_counts[0, v]
        id2word = dict([(self.cvectorizer.vocabulary_.get(w), w) for w in self.cvectorizer.vocabulary_])

        idx_sort = np.argsort(sum_counts_np)
        self.vocab = [id2word[idx_sort[cc]] for cc in range(v_size)]

        # Create dictionary and inverse dictionary
        self.word2id = dict([(w, j) for j, w in enumerate(self.vocab)])
        self.id2word = dict([(j, w) for j, w in enumerate(self.vocab)])

    def compose_samples(self, idx_permute: List[int], idx_range: Tuple[int, int]):
        sentence_list = []
        tags_list = []
        for idx in range(idx_range[0], idx_range[1]):
            raw_idx = idx_permute[idx]
            sentence = self.sentences[raw_idx]
            tags = self.tags_list[raw_idx]

            word_id_list = [self.word2id[w] for w in sentence.split() if w in self.word2id.keys()]
            if len(word_id_list) < 2:
                continue
            sentence_list.append(word_id_list)
            tags_list.append(tags)
        tags_list = self.mlb.transform(tags_list)
        return sentence_list, tags_list

    def split_train_test_valid(self):
        num_docs_tr = len(self.sentences)
        trSize = int(num_docs_tr * (1 - self.test_size_rate - self.valid_size_rate))
        vaSize = int(num_docs_tr * self.valid_size_rate)
        tsSize = int(num_docs_tr * self.test_size_rate)
        logging.info('number of train/valid/test: {}/{}/{}'.format(trSize, vaSize, tsSize))
        idx_permute = list(np.random.permutation(num_docs_tr))

        # Remove words not in train_data
        self.vocab = list(set(
            [w for idx_d in range(trSize)
             for w in self.sentences[idx_permute[idx_d]].split() if w in self.word2id.keys()]))
        self.word2id = dict([(w, j) for j, w in enumerate(self.vocab)])
        self.id2word = dict([(j, w) for j, w in enumerate(self.vocab)])

        docs_tr, tag_tr = self.compose_samples(idx_permute, (0, trSize))
        docs_va, tag_va = self.compose_samples(idx_permute, (trSize, trSize + vaSize))
        docs_ts, tag_ts = self.compose_samples(idx_permute, (trSize + vaSize, trSize + vaSize + tsSize))

        docs_ts_h1 = [[w for i, w in enumerate(doc) if i <= len(doc) / 2.0 - 1] for doc in docs_ts]
        docs_ts_h2 = [[w for i, w in enumerate(doc) if i > len(doc) / 2.0 - 1] for doc in docs_ts]
        tag_ts_h1, tag_ts_h2 = tag_ts, tag_ts

        return {'train': (docs_tr, tag_tr),
                'valid': (docs_va, tag_va),
                'test': (docs_ts, tag_ts),
                'test_h1': (docs_ts_h1, tag_ts_h1),
                'test_h2': (docs_ts_h2, tag_ts_h2),
                }

    def get_bow_representation(self, words_list: List[List[int]]):
        all_word_list = []
        doc_indices = []
        for doc_idx, words in enumerate(words_list):
            for w in words:
                all_word_list.append(w)
                doc_indices.append(doc_idx)
        bow = sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, all_word_list)),
                                shape=(len(words_list), len(self.vocab))).tocsr()
        return bow

    def save_to_file_bow(self, path_save=Path("mag_labeled")):
        path_save.mkdir(parents=True, exist_ok=True)

        self.build_vocabulary_and_label_transfer()
        with open(path_save / 'vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)

        for split_type, (docs, tags) in self.split_train_test_valid().items():
            n_docs = len(docs)
            bow = self.get_bow_representation(docs)

            # Split bow intro token/value pairs
            bow_tokens = [[w for w in bow[doc, :].indices] for doc in range(n_docs)]
            bow_counts = [[c for c in bow[doc, :].data] for doc in range(n_docs)]

            savemat(path_save / '{}_tokens.mat'.format(split_type),
                    {'tokens': bow_tokens}, do_compression=True)
            savemat(path_save / '{}_counts.mat'.format(split_type),
                    {'counts': bow_counts}, do_compression=True)
            savemat(path_save / '{}_tags.mat'.format(split_type),
                    {'tags': tags}, do_compression=True)


def main():
    with open('/mnt/conf_data/llda_data_by_year.pkl', 'rb') as f:
        llda_data_by_year = pickle.load(f)

    # for year, sentences_with_tags in tqdm(llda_data_by_year.items()):
    #     convertor = LabeledDataConvertor(sentences_with_tags, 0.02, 0.02)
    #     convertor.save_to_file_bow(Path('mag_labeled') / str(year))

    convertor = LabeledDataConvertor(llda_data_by_year[2019], 0.02, 0.1)
    convertor.save_to_file_bow(Path('mag_labeled') / str(2019))

    # all_sentences_with_tag = []
    # for year, sentences_with_tags in tqdm(llda_data_by_year.items()):
    #     all_sentences_with_tag += sentences_with_tags
    # convertor = LabeledDataConvertor(all_sentences_with_tag, 0.05, 0.05)
    # convertor.save_to_file_bow(Path('mag_labeled') / 'all')


if __name__ == '__main__':
    main()
