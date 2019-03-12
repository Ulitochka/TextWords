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


def get_word_id(word, word2id, id2word, max_vocab_size):
    if word not in word2id:
        if len(word2id) >= max_vocab_size:
            return word2id["<UNK>"]
        id2word.append(word)
        word2id[word] = len(word2id)
    return word2id[word]


def words_to_ids(words, word2id):
    ids = []
    for word in words:
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


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x
