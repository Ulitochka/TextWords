import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class CommandScorerModel(nn.Module):
    def __init__(self, input_size, hidden_size, device, verbose):
        super(CommandScorerModel, self).__init__()

        self.device = device

        self.verbose = verbose

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.cmd_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.state_gru = nn.GRU(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size, device=self.device)
        self.critic = nn.Linear(hidden_size, 1)
        self.att_cmd = nn.Linear(hidden_size * 2, 1)

    def forward(self, obs, commands, **kwargs):
        input_length = obs.size(0)
        batch_size = obs.size(1)
        nb_cmds = commands.size(2)

        # Network over the obs.
        embedded = self.embedding(obs)
        encoder_output, encoder_hidden = self.encoder_gru(embedded)
        state_output, state_hidden = self.state_gru(encoder_hidden, self.state_hidden)
        self.state_hidden = state_hidden
        value = self.critic(state_output)

        # Network over the commands.
        cmds_embedding = self.embedding.forward(commands)   # cmds_embedding torch.Size([3, 6, 17, 128])

        # Diff command for every batch element: cmds_encoding_last_states torch.Size([1, 3, 17, 128])
        _cmds_encoding_last_states = torch.stack([self.cmd_encoder_gru.forward(el)[-1] for el in cmds_embedding], 1)

        # Same observed state for all commands. torch.Size([1, 3, 54, 128])
        cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2)

        # Concatenate the observed state and command encodings.
        cmd_selector_input = torch.cat([cmd_selector_input, _cmds_encoding_last_states], dim=-1)

        # Compute one score per command.
        scores = F.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)  # 1 x Batch x cmds
        probs = F.softmax(scores, dim=2)  # 1 x Batch x cmds
        index = probs[0].multinomial(num_samples=1).unsqueeze(0)  # 1 x batch x indx

        if self.verbose:
            print('cmds_embedding', cmds_embedding.size())
            print('_cmds_encoding_last_states', _cmds_encoding_last_states.size())
            print('cmd_selector_input', cmd_selector_input.size())
            print('probs', probs.size())
            print('index', index.size())

        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
