import numpy as np
import torch
from typing import List, Dict, Any
from utils import preprocessing, text_process, get_word_id, words_to_ids, max_len, pad_sequences, to_pt


class TextPreprocessor:
    def __init__(self, tokenizer, device, word_vocab, single_word_verbs, EOS_id, preposition_map, word2id):
        self.tokenizer = tokenizer
        self.device = device
        self.word_vocab = word_vocab
        self.single_word_verbs = single_word_verbs
        self.EOS_id = EOS_id
        self.preposition_map = preposition_map
        self.word2id = word2id
        self.stop_tokens = ['', '-=']

    def get_game_step_info(self,
                           obs: List[str],
                           infos: Dict[str, List[Any]]
                           ):
        """
        Get all the available information, and concat them together to be tensor for
        a neural model. we use post padding here, all information are tokenized here.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.

        input_description = torch.Size([210, 3])
        commands = torch.Size([3, 6, 17])
        """

        # observations preprocessing

        inventory_token_list = [preprocessing(item, tokenizer=self.tokenizer) for item in infos["inventory"]]
        inventory_id_list = [words_to_ids(tokens, self.word2id, self.stop_tokens) for tokens in inventory_token_list]

        quest_token_list = [preprocessing(item, tokenizer=self.tokenizer) for item in infos["extra.recipe"]]
        quest_id_list = [words_to_ids(tokens, self.word2id, self.stop_tokens) for tokens in quest_token_list]

        description_token_list = [preprocessing(item, tokenizer=self.tokenizer) for item in infos["description"]]
        for i, d in enumerate(description_token_list):
            if len(d) == 0:
                description_token_list[i] = ["end"]  # if empty description, insert word "end"

        feedback_token_list = [preprocessing(item, str_type='feedback', tokenizer=self.tokenizer) for item in obs]
        feedback_id_list = [words_to_ids(tokens, self.word2id, self.stop_tokens) for tokens in feedback_token_list]

        description_id_token_list = [words_to_ids(tokens, self.word2id, self.stop_tokens) for tokens in description_token_list]
        description_id_list = [_d + _i + _q + _f for (_d, _i, _q, _f) in
                               zip(description_id_token_list,
                                   inventory_id_list,
                                   quest_id_list,
                                   feedback_id_list,
                                   )]

        input_description = pad_sequences(description_id_list,
                                          self.word2id,
                                          self.device,
                                          max_command_len=max_len(description_id_list))

        # command preprocessing
        admissible_commands_token_list = [[preprocessing(c, tokenizer=self.tokenizer) for c in c_list]
                                          for c_list in infos["admissible_commands"]]
        admissible_commands_id_list = [[words_to_ids(c, self.word2id, self.stop_tokens) for c in c_list]
                                       for c_list in admissible_commands_token_list]

        max_command_len = max([len(c) for c_list in admissible_commands_id_list for c in c_list])
        max_count_commands = max([len(c_list) for c_list in admissible_commands_id_list])

        commands = [pad_sequences(c_list, self.word2id, self.device, max_command_len=max_command_len,
                                  max_count_commands=max_count_commands)
                    for c_list in admissible_commands_id_list]
        commands = torch.stack(commands)

        return input_description, description_id_list, commands
