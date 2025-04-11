import torch
import torch.nn as nn
from modules.sentiment_analysis_circuits import construct_classification_circuit


class SentimentAnalysisXYZ(nn.Module):
    def __init__(self, encoder, args, p_weights=None):
        super().__init__()
        self.args = args
        n_qubits = args.num_latent + args.num_trash
        n_weights = (4 * n_qubits - 2) * args.depth
        self.encoder = encoder
        self.p_weights = (
            nn.Parameter(p_weights)
            if p_weights is not None
            else nn.Parameter(
                0.1 * torch.rand(n_weights * args.sentence_len), requires_grad=True
            )
        )

    def forward(self, features):
        if self.args.mode == 'train':
            predictions = torch.zeros(len(features))
        else:
            predictions = torch.zeros(len(features), 3)
        for count, document in enumerate(features):
            document_out = construct_classification_circuit(
                self.args, self.p_weights, document, self.encoder
            )
            predictions[count] = (
                ((document_out[0] + document_out[1] + document_out[2]) / 3) + 1
            ) / 2
        return predictions.to(torch.float32)
