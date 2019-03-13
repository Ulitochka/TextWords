import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class CommandScorerModel(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(CommandScorerModel, self).__init__()

        self.device = device

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
        nb_cmds = commands.size(1)

        embedded = self.embedding(obs)
        encoder_output, encoder_hidden = self.encoder_gru(embedded)

        state_output, state_hidden = self.state_gru(encoder_hidden, self.state_hidden)
        self.state_hidden = state_hidden
        value = self.critic(state_output)

        # Attention network over the commands.
        cmds_embedding = self.embedding.forward(commands)
        print('cmds_embedding', cmds_embedding.size())
        _, cmds_encoding_last_states = self.cmd_encoder_gru.forward(cmds_embedding)  # 1 x cmds x hidden

        # Same observed state for all commands.
        cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden

        # Same command choices for the whole batch.
        cmds_encoding_last_states = torch.stack([cmds_encoding_last_states] * batch_size, 1)  # 1 x batch x cmds x hidden

        # Concatenate the observed state and command encodings.
        cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoding_last_states], dim=-1)

        # Compute one score per command.
        scores = F.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)  # 1 x Batch x cmds

        probs = F.softmax(scores, dim=2)  # 1 x Batch x cmds
        index = probs[0].multinomial(num_samples=1).unsqueeze(0)  # 1 x batch x indx
        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
