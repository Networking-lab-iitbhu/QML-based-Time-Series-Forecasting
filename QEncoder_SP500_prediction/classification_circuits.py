import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pennylane as qml
from pennylane import numpy as np
import torch


# from https://arxiv.org/pdf/1905.10876.pdf
def circuit_7(weights, args):
    weights = torch.flatten(weights)
    depth = args.depth
    n_qubits = args.num_latent + args.num_trash
    w_count = 0
    weights = torch.flatten(weights)
    for j in range(depth):
        for i in range(n_qubits):
            qml.RX(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(0,n_qubits-1,2):
            qml.CRZ(weights[w_count],[i+1, i])
            w_count += 1
        for i in range(n_qubits):
            qml.RX(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(1,n_qubits-1,2):
            qml.CRZ(weights[w_count],[i+1, i])
            w_count += 1


def autoencoder_circuit_trained(weights, args, features=None):
    weights = weights.reshape(-1, args.num_latent + args.num_trash)
    qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))



def keep_feature_from_padding(args, feature, time_step):
    
    # feature = one of the five attributes [op,hi,cl,lo,vol]
    
    is_padded = np.all(np.isclose(feature, feature[0])) 
    
    #is_padded checks if all the 10 timestamps values are same.
    
    #It checks the data of all other days with the day 1 data.
    
    if is_padded:
        #if all the 10 timestamps have same values its constant data or just some dummy values and they wont provide any useful info.
        #so dont use this feature at all.
        return False 
                      
    
    return True


def construct_classification_circuit(args, weights, features, trained_encoder=None):
    
    sentiment_dev=qml.device("default.qubit", wires=args.num_latent + 2* args.num_trash + 1)
    

    if args.mode =='train':
        qnode = qml.qnode(sentiment_dev, interface="torch", diff_method="backprop")
    

    @qnode
    def classification_circuit(args, model_weights, features, trained_encoder=None):
        num_latent, num_trash = args.num_latent, args.num_trash
        
        e_weights = torch.tensor(trained_encoder.weights.detach(), requires_grad=False)
            
        model_weights = model_weights.reshape(args.n_cells, -1)
        
        
       

        #features here means 10 day data for all 5 attributes.
        
        #features = [[op1,op2,...,op10],[cl1,cl2,..cl10],[hi1,hi2...hi10],[lo1,lo2,..lo10],[vol1,vol2...vol10]]
        
        
        # Encode First Feature
        inital_feature = features[0] #first feature , feature = an array of size 10, 10 timestamps.


        qml.AngleEmbedding(
            inital_feature[: num_trash + num_latent],
            wires=range(num_latent + num_trash),
            rotation="X",
        )
        #qml.AngleEmbedding(
        #    inital_feature[num_trash + num_latent: (2 * num_trash + num_latent)+2],
        #    wires=range(num_latent + num_trash+1),
        #    rotation="Y",
        #)
        
        circuit_7(model_weights[0], args)

        for i in range(1, len(features)):  #this loop runs from i=1 to i=4, [len(features)=5]
                                      
            if keep_feature_from_padding(args, features[i], i):
                #features[i] = ith feature from [op,hi,lo,cl,vol]
                
                    # Compress the features
                autoencoder_circuit_trained(e_weights, args)                 
                    # Embed the new feature on the freed up qubits
                    
                qml.AngleEmbedding(
                        features[i][: args.num_trash], wires=range(num_trash), rotation="X"
                )
                qml.AngleEmbedding(
                        features[i][num_trash : 2 * (num_trash)],
                        wires=range(num_trash),
                        rotation="Y",
                )
                circuit_7(model_weights[i], args)

        #since above loop runs for four times 
        #1st time = compress the data which contains 0th feature and then inject 1st feature and process it
        #2nd time = compress the data again which contains 0,1 features and then inject 2nd feature and process
        #.. 4th time = compress data containing 0,1,2,3 and then inject 5th feaure and process
        
        circuit_7(model_weights[-1], args)  #now last time just process all the data at once , final quantum layer applied before measurement
        # Measure the output
        if args.mode == "train":
            return qml.probs(0)
    
    return classification_circuit(args, weights, features, trained_encoder)
