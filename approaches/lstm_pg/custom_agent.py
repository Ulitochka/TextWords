import os
import yaml
import time
import uuid
import copy
import warnings
from collections import defaultdict
from typing import List, Dict, Any, Optional

from numpy.random import RandomState
import numpy as np
import spacy
from textworld import EnvInfos
import torch

from utils import preprocessing, text_process, get_word_id
from model import CommandScorerModel
from text_preprocessor import TextPreprocessor


class CustomAgent:

    def __init__(self) -> None:

        self.model_id = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8]

        self.update_frequency = 10
        self.log_frequency = 1000
        self.gamma = 0.9

        self.use_cuda = False
        self.device = 'cpu'

        # load config & vocab
        with open("./vocab.txt") as f:
            self.word_vocab = f.read().split("\n")
        with open("config.yaml") as reader:
            self.config = yaml.safe_load(reader)

        self.max_vocab_size = len(self.word_vocab)
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        self.EOS_id = self.word2id["</S>"]

        # Set the random seed manually for reproducibility.
        np.random.seed(self.config['general']['random_seed'])
        torch.manual_seed(self.config['general']['random_seed'])
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.config['general']['random_seed'])
                self.use_cuda = True
                self.device = 'cuda:0'
        else:
            self.use_cuda = False

        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.nb_epochs = self.config['training']['nb_epochs']
        self.experiment_tag = self.config['checkpoint']['experiment_tag']
        self.model_checkpoint_path = self.config['checkpoint']['model_checkpoint_path']
        self.save_frequency = self.config['checkpoint']['save_frequency']
        self.update_per_k_game_steps = self.config['general']['update_per_k_game_steps']    # update_frequency ?
        self.clip_grad_norm = self.config['training']['optimizer']['clip_grad_norm']

        self._initialized = False
        self._epsiode_has_started = False
        self.current_episode = 0
        self.best_avg_score_so_far = 0.0

        # model_init
        self.model = CommandScorerModel(input_size=self.max_vocab_size,
                                        hidden_size=128,
                                        device=self.device,
                                        verbose=False)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.config['training']['optimizer']['learning_rate'])
        self.model.to(self.device)

        # using checkpoint
        if self.config['checkpoint']['load_pretrained']:
            self.load_pretrained_model(
                self.model_checkpoint_path + '/' + self.config['checkpoint']['pretrained_experiment_tag'] + '.pt')
        if self.use_cuda:
            self.model.cuda()

        # tokenizer load
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        self.preposition_map = {"take": "from",
                                "chop": "with",
                                "slice": "with",
                                "dice": "with",
                                "cook": "with",
                                "insert": "into",
                                "put": "on"}
        self.single_word_verbs = set(["inventory", "look"])

        self.mode = "test"

        # TODO
        self.rng = RandomState()
        self.text_processor = TextPreprocessor(self.nlp,
                                               self.device,
                                               self.word_vocab,
                                               self.single_word_verbs,
                                               self.EOS_id,
                                               self.preposition_map,
                                               self.word2id)

    def infos_to_request(self) -> EnvInfos:
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.extras = ["recipe"]
        return request_infos

    def tokenize(self, text):
        text = preprocessing(text, tokenizer=self.nlp)
        word_ids = [get_word_id(t, self.word2id, self.max_vocab_size) for t in text]
        return word_ids

    def discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.gamma * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def select_additional_infos(self) -> EnvInfos:
        return EnvInfos(description=True,
                        inventory=True,
                        admissible_commands=True,
                        has_won=True,
                        extras=["recipe"],
                        has_lost=True)

    def load_pretrained_model(self, load_from):
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                state_dict = torch.load(load_from)
            else:
                state_dict = torch.load(load_from, map_location='cpu')
            self.model.load_state_dict(state_dict)
        except:
            print("Failed to load checkpoint...")

    def finish(self) -> None:
        """
        All games in the batch are finished. One can choose to save checkpoints,
        evaluate on validation set, or do parameter annealing here.

        """
        # Game has finished (either win, lose, or exhausted all the given steps).

        self.final_rewards = np.array(self.scores[-1], dtype='float32')  # batch
        dones = []
        for d in self.dones:
            d = np.array([float(dd) for dd in d], dtype='float32')
            dones.append(d)
        dones = np.array(dones)
        step_used = 1.0 - dones
        self.step_used_before_done = np.sum(step_used, 0)  # batch

        # save checkpoint
        if self.mode == "train" and self.current_episode % self.save_frequency == 0:
            avg_score = np.mean(self.final_rewards)
            if avg_score > self.best_avg_score_so_far:
                self.best_avg_score_so_far = avg_score

                save_to = self.model_checkpoint_path + '/' + self.experiment_tag + "_episode_" + str(
                    self.current_episode) + ".pt"
                if not os.path.isdir(self.model_checkpoint_path):
                    os.mkdir(self.model_checkpoint_path)
                torch.save(self.model.state_dict(), save_to)
                print("========= saved checkpoint =========")

        self.current_episode += 1

    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0

        self.dones = []
        self.scores = []

    def eval(self):
        self.mode = "test"
        self.model.reset_hidden(1)

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> Optional[
        List[str]]:

        input_tensor, _, commands_tensor = self.text_processor.get_game_step_info(obs, infos)
        outputs, indexes, values = self.model(input_tensor, commands_tensor)

        print('outputs:', outputs)
        print('indexes:', indexes[0])
        print('values:', values)

        actions_per_batch = []
        for cmds_i in range(self.batch_size):
            action = None
            try:
                action = infos["admissible_commands"][cmds_i][indexes[0][cmds_i]]
            except IndexError:
                # TODO torch.Size([3, max_seq_len, max_commands_number])
                action = self.rng.choice(infos["admissible_commands"][cmds_i])
                warnings.warn("Warning model choice padded array: %s" % (
                        str(infos["admissible_commands"][cmds_i]) + ' ' +
                        str(len(infos["admissible_commands"][cmds_i])) + ' '+ str(indexes[0][cmds_i]),))
            actions_per_batch.append(action)
        print('*' * 100)

        if self.mode == "eval":
            if all(dones):
                self.model.reset_hidden(1)
            return actions_per_batch

        self.no_train_step += 1

        if self.transitions:
            reward = score - self.last_score  # Reward is the gain/loss in score.
            self.last_score = score
            if infos["has_won"]:
                reward += 100
            if infos["has_lost"]:
                reward -= 100

            self.transitions[-1][0] = reward  # Update reward information.




        if self.mode == "eval":
            if done:
                self.model.reset_hidden(1)
            return action

        if not self._epsiode_has_started:
            self.start_episode(obs, infos)

        if all(dones):
            self.end_episode(obs, scores, infos)
            return  # Nothing to return.

        if self.current_step > 0:
            # append scores / dones from previous step into memory
            self.scores.append(scores)
            self.dones.append(dones)



        if all(dones):
            self.end_episode(obs, scores, infos)
            return  # Nothing to return.

        return [self.rng.choice(cmds) for cmds in infos["admissible_commands"]]
