import torch
import torch.nn as nn

class SentimentAnalysisBidirectionalLSTMTemperature(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, pretrained_embedding, init_temp=1.0):
        super(SentimentAnalysisBidirectionalLSTMTemperature, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 3)

        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.temperature_scale(out)

        return out, hidden
        
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros((self.n_layers * 2, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.n_layers * 2, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
