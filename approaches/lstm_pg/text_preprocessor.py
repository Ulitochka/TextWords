from typing import List, Dict, Any
from utils import preprocessing, text_process, get_word_id, words_to_ids, max_len, pad_sequences, to_pt


class TextPreprocessor:
    def __init__(self, tokenizer, use_cuda, word_vocab, single_word_verbs, EOS_id, preposition_map, word2id):
        self.tokenizer = tokenizer
        self.use_cuda = use_cuda
        self.word_vocab = word_vocab
        self.single_word_verbs = single_word_verbs
        self.EOS_id = EOS_id
        self.preposition_map = preposition_map
        self.word2id = word2id

    def get_game_step_info(self,
                           obs: List[str],
                           infos: Dict[str, List[Any]],
                           prev_actions):
        """
        Get all the available information, and concat them together to be tensor for
        a neural model. we use post padding here, all information are tokenized here.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        inventory_token_list = [preprocessing(item, tokenizer=self.tokenizer) for item in infos["inventory"]]
        inventory_id_list = [words_to_ids(tokens, self.word2id) for tokens in inventory_token_list]

        feedback_token_list = [preprocessing(item, str_type='feedback', tokenizer=self.tokenizer) for item in obs]
        feedback_id_list = [words_to_ids(tokens, self.word2id) for tokens in feedback_token_list]

        quest_token_list = [preprocessing(item, tokenizer=self.tokenizer) for item in infos["extra.recipe"]]
        quest_id_list = [words_to_ids(tokens, self.word2id) for tokens in quest_token_list]

        prev_action_token_list = [preprocessing(item, tokenizer=self.tokenizer) for item in prev_actions]
        prev_action_id_list = [words_to_ids(tokens, self.word2id) for tokens in prev_action_token_list]

        description_token_list = [preprocessing(item, tokenizer=self.tokenizer) for item in infos["description"]]
        for i, d in enumerate(description_token_list):
            if len(d) == 0:
                description_token_list[i] = ["end"]  # if empty description, insert word "end"

        description_id_list = [words_to_ids(tokens, self.word2id) for tokens in description_token_list]
        description_id_list = [_d + _i + _q + _f + _pa for (_d, _i, _q, _f, _pa) in
                               zip(description_id_list, inventory_id_list, quest_id_list, feedback_id_list,
                                   prev_action_id_list)]

        input_description = pad_sequences(description_id_list, maxlen=max_len(description_id_list)).astype('int32')
        input_description = to_pt(input_description, self.use_cuda)

        return input_description, description_id_list

    def word_ids_to_commands(self, verb, adj, noun, adj_2, noun_2):
        """
        Turn the 5 indices into actual command strings.

        Arguments:
            verb: Index of the guessing verb in vocabulary
            adj: Index of the guessing adjective in vocabulary
            noun: Index of the guessing noun in vocabulary
            adj_2: Index of the second guessing adjective in vocabulary
            noun_2: Index of the second guessing noun in vocabulary
        """
        # turns 5 indices into actual command strings
        if self.word_vocab[verb] in self.single_word_verbs:
            return self.word_vocab[verb]
        if adj == self.EOS_id:
            res = self.word_vocab[verb] + " " + self.word_vocab[noun]
        else:
            res = self.word_vocab[verb] + " " + self.word_vocab[adj] + " " + self.word_vocab[noun]
        if self.word_vocab[verb] not in self.preposition_map:
            return res
        if noun_2 == self.EOS_id:
            return res
        prep = self.preposition_map[self.word_vocab[verb]]
        if adj_2 == self.EOS_id:
            res = res + " " + prep + " " + self.word_vocab[noun_2]
        else:
            res = res + " " + prep + " " + self.word_vocab[adj_2] + " " + self.word_vocab[noun_2]
        return res
