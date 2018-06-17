# -*- coding: utf-8 -*-

# ------------------------------------
# Create On 2018/6/2 14:49 
# File Name: proces_data.py
# Edit Author: lnest
# ------------------------------------
import json
import random
from itertools import permutations

from time_eval import time_count

cut_off_length = 8
max_legnth = 10


def _read_dataset(file_name):
    words = []
    with open(file_name, 'r') as rh:
        lines = rh.readlines()
        r_idx = 0
        word_scope = False
        word = []
        last_flag = ''  # record last flag
        while r_idx < len(lines):
            contnts = lines[r_idx].strip().split()

            if len(contnts) != 2:
                word_scope = False
                r_idx += 1
                continue

            startswith_o = contnts[1].startswith('O')
            startswith_b = contnts[1].startswith('B')
            last_startswith_i = last_flag.startswith('I')
            if startswith_o:
                word_scope = False
                if len(word) > 2:
                    cut_off = min(cut_off_length, len(word))
                    words.append(''.join(word[:cut_off]))
                word = []
                r_idx += 1
                continue
            elif startswith_b:
                word_scope = True
            # handle sequence such as 'B I I B I I O'
            if startswith_b and last_startswith_i:
                if len(word) > 2:
                    words.append(''.join(word))
                word = []

            if word_scope:
                word.append(contnts[0])

            last_flag = contnts[1]
            r_idx += 1

    return words


def generate_wordmap(corpus, word_map_file, ratio=0.9):
    word2idx = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
    idx2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
    word_cnt = len(word2idx)

    with open(corpus, 'r') as fr:
        lines = fr.readlines()
        rows = len(lines)
        for line in lines[:int(rows*ratio)]:
            words = list(line)
            for word in words:
                if word not in word2idx:
                    word2idx[word] = word_cnt
                    idx2word[word_cnt] = word
                    word_cnt += 1
    word_map = {'word2idx': word2idx, 'idx2word': idx2word}
    json.dump(word_map, open(word_map_file, 'w'), ensure_ascii=False)


@time_count
def process_data(input_file, output_file):
    """ read single or multiple file and write target contents into output_file
    :param input_file: 
    :param output_file: 
    :return: 
    """
    with open(output_file, 'w') as wh:
        if isinstance(input_file, list):
            for _file_name in input_file:
                words = _read_dataset(_file_name)
                if words:
                    wh.write('\n'.join(words))
                else:
                    print('Get empty contents in file {}'.format(_file_name))
        else:
            words = _read_dataset(input_file)
            if words:
                wh.write('\n'.join(words))
            else:
                print('Get empty contents in file {}'.format(input_file))
    print('done')


def sentence2id(sentence, word_map):
    word2id = word_map['word2idx']
    sentence_index = [str(word2id.get(word, '3')) for word in sentence.strip()]
    return sentence_index


def add_pad(ids, uniform_size):
    if len(ids) < uniform_size:
        for i in range(uniform_size - len(ids)):
            ids.append('0')
    else:
        ids = ids[:uniform_size]
    return ids


@time_count
def write_data2file(data, file_handler, dict_size, max_len=9):
    twh_buffer = []
    for row_id, line in enumerate(data, 1):
        permut_clip_size = 10
        noise_frequence = 3  # add noise for every three line
        # drop word which length is bigger than max_len
        if len(line) > max_len:
            continue
        for permut_id, permut in enumerate(permutations(line, len(line)), 0):
            if permut_id >= permut_clip_size:
                break
            permut = list(permut)
            if permut_id % noise_frequence == 0:
                random_pos = random.choice(range(len(permut)))
                # the index of first valid word is 4
                random_word_idx = random.choice(range(4, dict_size))
                permut[random_pos] = str(random_word_idx)
            add_pad(permut, 9)
            add_pad(line, 9)
            sentence_ids = ' '.join(permut) + '\t' + ' '.join(line)
            twh_buffer.append(sentence_ids)
        if row_id % 10000 == 0:
            if twh_buffer:
                file_handler.write('\n'.join(twh_buffer))
                print('write %d line into %s' % (row_id, file_handler))
                twh_buffer = ['']
    if len(twh_buffer) > 1:
        file_handler.write('\n'.join(twh_buffer))
    print('done')


@time_count
def get_test_and_train(dataset, train_file, test_file, word_map_file, ratio):
    word_map = json.load(open(word_map_file))
    with open(dataset, 'r') as rh:
        lines = rh.readlines()
        assert 0 < ratio < 1
        if ratio < 0.5:
            ratio = 1 - ratio
        pivot = int(len(lines) * ratio)
        train_scope = lines[:pivot]
        test_scope = lines[pivot:]
        train_data = [sentence2id(sentence, word_map) for sentence in train_scope]
        test_data = [sentence2id(sentence, word_map) for sentence in test_scope]
    idx2word = word_map['idx2word']
    dict_size = len(idx2word)
    with open(train_file, 'w') as twh:
        write_data2file(train_data, twh, dict_size)

    with open(test_file, 'w') as test_wh:
        write_data2file(test_data, test_wh, dict_size)

    print('done')


if __name__ == '__main__':
    process_data(['../data/example.dev', '../data/example.test', '../data/example.train'], '../data/data_all')
    generate_wordmap('../data/data_all', '../data/my_word_map.json')
    get_test_and_train('../data/data_all', '../data/train', '../data/test', '../data/my_word_map.json', 0.1)
