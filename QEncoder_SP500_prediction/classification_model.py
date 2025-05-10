import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from .classification_circuits import construct_classification_circuit


class Classifier(nn.Module):
    def __init__(self, encoder, args, model_weights=None):
        super().__init__()
        self.args = args
        n_qubits = args.num_latent + args.num_trash
        #n_weights = (4 * n_qubits - 2) * args.depth
        n_weights = ((4 * n_qubits - 2) * args.depth)*4
        self.encoder = encoder
        self.model_weights = (
            nn.Parameter(model_weights)
            if model_weights is not None
            else nn.Parameter(
                0.1 * torch.rand(n_weights * args.n_cells), requires_grad=True
            )
        )

    def forward(self, features): #features is a 256 size 5*10 matrix , like all windows of five attributes values for 10 days 
        if self.args.mode == 'train':
            predictions = torch.zeros(len(features))
            
        for count, document in enumerate(features):
            document_out = construct_classification_circuit(
                self.args, self.model_weights, document, self.encoder #document = one window of 10 days data for all 5 attributes [op,hi,lo,cl,vol]
            )
            if self.args.mode == 'train':
                predictions[count] = document_out[0]
            #document_out will return probability of 0th qubit i.e [a,b] 
            #a = probability it collapses to 0, b =probability it collapses to 1. 
            
        return predictions.to(torch.float32)
