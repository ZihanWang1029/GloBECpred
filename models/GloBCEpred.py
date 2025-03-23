import torch
import torch.nn as nn

class GloBCEpred(nn.Module):
    def __init__(self, input_size=51, conv_filters=128, gru_hidden_size=128, output_size=1, num_heads=8):
        super(GloBCEpred, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_filters, kernel_size=5, stride=1)
        self.batch_norm = nn.BatchNorm1d(conv_filters)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.6)

        self.gru = nn.GRU(input_size=conv_filters, hidden_size=gru_hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.multihead_attention = nn.MultiheadAttention(embed_dim=gru_hidden_size*2, num_heads=num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(gru_hidden_size*2)

        self.fc_attention = nn.Linear(gru_hidden_size * 2, 10)
        self.fc_output = nn.Linear(10, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1d(x))
        x = self.batch_norm(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)

        attn_output, _ = self.multihead_attention(x, x, x)

        x = x + attn_output
        x = self.norm1(x)

        x = torch.sigmoid(self.fc_attention(x.mean(dim=1)))
        x = self.fc_output(x)

        return x.squeeze(-1)