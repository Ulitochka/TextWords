import torch
import numpy as np


def preprocessing(s, str_type='None', tokenizer=None, lower_case=True):
    if s is None:
        return ["nothing"]
    s = s.replace("\n", ' ')
    if s.strip() == "":
        return ["nothing"]
    if str_type == 'feedback':
        if "$$$$$$$" in s:
            s = ""
        if "-=" in s:
            s = s.split("-=")[0]
    s = s.strip()
    if len(s) == 0:
        return ["nothing"]
    tokens = [t.text for t in tokenizer(s)]
    if lower_case:
        tokens = [t.lower() for t in tokens]
    return tokens


def text_process(self, texts):
    texts = list(map(self._tokenize, texts))
    max_len = max(len(l) for l in texts)
    padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

    for i, text in enumerate(texts):
        padded[i, :len(text)] = text

    padded_tensor = torch.from_numpy(padded).type(torch.long).to(self.device)
    padded_tensor = padded_tensor.permute(1, 0)     # Batch x Seq => Seq x Batch
    return padded_tensor


def get_word_id(word, word2id, max_vocab_size):
    if word not in word2id:
        if len(word2id) >= max_vocab_size:
            return word2id["<UNK>"]
        word2id[word] = len(word2id)
    return word2id[word]


def words_to_ids(words, word2id, stop_tokens):
    ids = []
    for word in words:
        if word.strip() not in stop_tokens:
            try:
                ids.append(word2id[word])
            except KeyError:
                ids.append(1)
    return ids


def max_len(list_of_list):
    return max(map(len, list_of_list))


def to_pt(np_matrix, enable_cuda=False, type='long'):
    if type == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor))
    elif type == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor))


def pad_sequences(sequences, word2id, device, max_command_len=None, max_count_commands=None, dtype='int32', value=0.):

    if max_count_commands:
        padded = np.ones((max_count_commands, max_command_len)) * word2id["<PAD>"]
    else:
        padded = np.ones((len(sequences), max_command_len)) * word2id["<PAD>"]

    for i, text in enumerate(sequences):
        padded[i, :len(text)] = text

    padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
    padded_tensor = padded_tensor.permute(1, 0)     # Batch x Seq => Seq x Batch
    return padded_tensor

