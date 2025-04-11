import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import pennylane as qml
from pennylane import numpy as np
from QEncoder_SP500_prediction.device_router import route_device 
# Pytorch imports
import torch
from QEncoder_SP500_prediction.modules.layers import p, autoencoder_circuit_no_swap  # Absolute Import



def keep_feature_from_padding(args, feature, time_step):
    is_padded = np.all(np.isclose(feature, feature[0]))
    pad_to = (args.version % 5) * 4
    if is_padded and time_step > pad_to:
        return False
    return True
def construct_classification_circuit(args, weights, features, trained_encoder=None):
    sentiment_dev = route_device(args,'medium')
    #sentiment_dev = route_device(args, 0.005)

    if args.mode =='train':
        qnode = qml.qnode(sentiment_dev, interface="torch", diff_method="backprop")
    else:
        qnode = qml.qnode(sentiment_dev, interface="torch")

    @qnode
    def classification_circuit(args, p_weights, features, trained_encoder=None):
        num_latent, num_trash = args.num_latent, args.num_trash
        if args.model not in ["ablation_angle","ablation_angle_amp"]:
            e_weights = torch.tensor(trained_encoder.weights.detach(), requires_grad=False)
        p_weights = p_weights.reshape(args.sentence_len, -1)


        # Encode First Feature
        inital_feature = features[0]


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
        p(p_weights[0], args)

        for i in range(1, len(features)):
            if args.model == "ablation_angle":
                # emb_wires = int(np.ceil(np.log2(2*num_trash)))
                # state = features[i][: (2 * num_trash)]
                # state = state / np.linalg.norm(state)
                # padded_state = np.zeros(2**emb_wires)
                # padded_state[:state.shape[0]] = state
                # if not np.isnan(state[0]):
                #     qml.MottonenStatePreparation(padded_state, wires=range(emb_wires))
                # p(p_weights[i], args)
                qml.AngleEmbedding(
                        features[i][: args.num_trash], wires=range(num_trash), rotation="X"
                    )
                qml.AngleEmbedding(
                        features[i][num_trash : 2 * (num_trash)],
                        wires=range(num_trash),
                        rotation="Y",
                    )
                p(p_weights[i], args)
            elif args.model == "ablation_angle_amp":
                emb_wires = int(np.ceil(np.log2(2*num_trash)))
                state = features[i][: (2 * num_trash)]
                state = state / np.linalg.norm(state)
                padded_state = np.zeros(2**emb_wires)
                padded_state[:state.shape[0]] = state
                if not np.isnan(state[0]):
                    qml.MottonenStatePreparation(padded_state, wires=range(emb_wires))
                p(p_weights[i], args)
            else:
                if args.pad_mode !='selective' or keep_feature_from_padding(args, features[i], i):
                    # Compress the features
                    autoencoder_circuit_no_swap(e_weights, args)
                    # Embed the new word on the freed up qubits
                    qml.AngleEmbedding(
                        features[i][: args.num_trash], wires=range(num_trash), rotation="X"
                    )
                    qml.AngleEmbedding(
                        features[i][num_trash : 2 * (num_trash)],
                        wires=range(num_trash),
                        rotation="Y",
                    )
                    p(p_weights[i], args)


        p(p_weights[-1], args)
        # Measure the output
        if args.mode == "train":
            return qml.probs(0)
        else:
            return [
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            ]

    @qnode
    def classification_circuit_amp_mean(args, p_weights, features):
        features = np.mean(features, axis=0)
        p_weights = p_weights.reshape(3, -1)
        qml.AmplitudeEmbedding(
            features,
            wires=range(args.num_latent + args.num_trash),
            pad_with=0,
            normalize=True,
        )
        for i in range(1):
            p(p_weights[i], args)
        if args.mode == "train":
            return qml.probs(0)
        else:
            return [
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            ]

    @qnode
    def classification_circuit_different_combine_pairs(
        args, p_weights, features, trained_encoder
    ):
        num_latent, num_trash = args.num_latent, args.num_trash
        e_weights = torch.tensor(trained_encoder.weights.detach(), requires_grad=False)
        p_weights = p_weights.reshape(args.sentence_len, -1)

        # Encode First Feature
        inital_feature = features[0]
        second_feature = features[1]


        qml.AngleEmbedding(
            inital_feature[: num_trash + num_latent],
            wires=range(num_latent + num_trash),
            rotation="X",
        )
        qml.AngleEmbedding(
            second_feature[: num_trash + num_latent],
            wires=range(num_latent + num_trash),
            rotation="Y",
        )
        p(p_weights[0], args)
        for i in range(2, len(features), 2):
            autoencoder_circuit_no_swap(e_weights, args)
            qml.AngleEmbedding(
                features[i][: args.num_trash], wires=range(num_trash), rotation="X"
            )

            if i != len(features)-1:
                qml.AngleEmbedding(
                features[i+1][: args.num_trash], wires=range(num_trash), rotation="Y"
            )
            p(p_weights[i], args)
        p(p_weights[-1], args)
        # Measure the output
        if args.mode == "train":
            return qml.probs(0)
        else:
            return [
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            ]

    if args.model == "amp_mean":
        return classification_circuit_amp_mean(args, weights, features)
    elif args.model == "pair_encoding":
        return classification_circuit_different_combine_pairs(args, weights, features, trained_encoder)
    else:
        # drawer = qml.draw(classification_circuit, show_all_wires=True, wire_order=[2,1,0,3])
        # print(drawer(args, weights, features, trained_encoder))
        # exit()
        # specs_func = qml.specs(classification_circuit)
        # x = specs_func(args, weights, features, trained_encoder)
        # gates = x['resources'].num_gates
        # num_ent_layers = x['resources'].gate_types['BasicEntanglerLayers']

        # num_gates = (gates - num_ent_layers) + (num_ent_layers*14*args.depth)
        # with open(f'{args.dataset}_{args.depth}_{args.model}_gates.txt', 'a') as f:
        #     f.write(str(num_gates) + '\n')
        return classification_circuit(args, weights, features, trained_encoder)
# import pennylane as qml
# import torch
# from QEncoder_SP500_prediction.device_router import route_device
# from QEncoder_SP500_prediction.modules.layers import p, autoencoder_circuit_no_swap  # Absolute Import

# def keep_feature_from_padding(args, feature, time_step):
#     is_padded = np.all(np.isclose(feature, feature[0]))
#     pad_to = (args.version % 5) * 4
#     if is_padded and time_step > pad_to:
#         return False
#     return True

# def construct_classification_circuit(args, weights, features, trained_encoder=None):
#     # Unpacking the tuple returned by route_device (device, noise_transform)
#     sentiment_dev, insert_noise = route_device(args, 'medium')  # Extract both device and noise transform

#     if args.mode == 'train':
#         qnode = qml.qnode(sentiment_dev, interface="torch", diff_method="backprop")
#     else:
#         qnode = qml.qnode(sentiment_dev, interface="torch")

#     @qnode
#     @insert_noise  # Apply noise transformation
#     def classification_circuit(args, p_weights, features, trained_encoder=None):
#         num_latent, num_trash = args.num_latent, args.num_trash
#         if args.model not in ["ablation_angle", "ablation_angle_amp"]:
#             e_weights = torch.tensor(trained_encoder.weights.detach(), requires_grad=False)
#         p_weights = p_weights.reshape(args.sentence_len, -1)

#         # Encode First Feature
#         initial_feature = features[0]

#         qml.AngleEmbedding(
#             initial_feature[: num_trash + num_latent],
#             wires=range(num_latent + num_trash),
#             rotation="X",
#         )
        
#         p(p_weights[0], args)

#         for i in range(1, len(features)):
#             if args.model == "ablation_angle":
#                 qml.AngleEmbedding(
#                     features[i][: args.num_trash], wires=range(num_trash), rotation="X"
#                 )
#                 qml.AngleEmbedding(
#                     features[i][num_trash : 2 * (num_trash)],
#                     wires=range(num_trash),
#                     rotation="Y",
#                 )
#                 p(p_weights[i], args)
#             elif args.model == "ablation_angle_amp":
#                 emb_wires = int(np.ceil(np.log2(2 * num_trash)))
#                 state = features[i][: (2 * num_trash)]
#                 state = state / np.linalg.norm(state)
#                 padded_state = np.zeros(2**emb_wires)
#                 padded_state[:state.shape[0]] = state
#                 if not np.isnan(state[0]):
#                     qml.MottonenStatePreparation(padded_state, wires=range(emb_wires))
#                 p(p_weights[i], args)
#             else:
#                 if args.pad_mode != 'selective' or keep_feature_from_padding(args, features[i], i):
#                     # Compress the features
#                     autoencoder_circuit_no_swap(e_weights, args)
#                     # Embed the new word on the freed-up qubits
#                     qml.AngleEmbedding(
#                         features[i][: args.num_trash], wires=range(num_trash), rotation="X"
#                     )
#                     qml.AngleEmbedding(
#                         features[i][num_trash : 2 * (num_trash)],
#                         wires=range(num_trash),
#                         rotation="Y",
#                     )
#                     p(p_weights[i], args)

#         p(p_weights[-1], args)

#         # Measure the output
#         if args.mode == "train":
#             return qml.probs(0)
#         else:
#             return [
#                 qml.expval(qml.PauliX(0)),
#                 qml.expval(qml.PauliY(0)),
#                 qml.expval(qml.PauliZ(0)),
#             ]

#     @qnode
#     @noise_transform  # Apply noise transformation
#     def classification_circuit_amp_mean(args, p_weights, features):
#         features = np.mean(features, axis=0)
#         p_weights = p_weights.reshape(3, -1)
#         qml.AmplitudeEmbedding(
#             features,
#             wires=range(args.num_latent + args.num_trash),
#             pad_with=0,
#             normalize=True,
#         )
#         for i in range(1):
#             p(p_weights[i], args)
#         if args.mode == "train":
#             return qml.probs(0)
#         else:
#             return [
#                 qml.expval(qml.PauliX(0)),
#                 qml.expval(qml.PauliY(0)),
#                 qml.expval(qml.PauliZ(0)),
#             ]

#     @qnode
#     @noise_transform  # Apply noise transformation
#     def classification_circuit_different_combine_pairs(
#         args, p_weights, features, trained_encoder
#     ):
#         num_latent, num_trash = args.num_latent, args.num_trash
#         e_weights = torch.tensor(trained_encoder.weights.detach(), requires_grad=False)
#         p_weights = p_weights.reshape(args.sentence_len, -1)

#         # Encode First Feature
#         initial_feature = features[0]
#         second_feature = features[1]

#         qml.AngleEmbedding(
#             initial_feature[: num_trash + num_latent],
#             wires=range(num_latent + num_trash),
#             rotation="X",
#         )
#         qml.AngleEmbedding(
#             second_feature[: num_trash + num_latent],
#             wires=range(num_latent + num_trash),
#             rotation="Y",
#         )
#         p(p_weights[0], args)
#         for i in range(2, len(features), 2):
#             autoencoder_circuit_no_swap(e_weights, args)
#             qml.AngleEmbedding(
#                 features[i][: args.num_trash], wires=range(num_trash), rotation="X"
#             )

#             if i != len(features) - 1:
#                 qml.AngleEmbedding(
#                     features[i + 1][: args.num_trash], wires=range(num_trash), rotation="Y"
#                 )
#             p(p_weights[i], args)
#         p(p_weights[-1], args)

#         # Measure the output
#         if args.mode == "train":
#             return qml.probs(0)
#         else:
#             return [
#                 qml.expval(qml.PauliX(0)),
#                 qml.expval(qml.PauliY(0)),
#                 qml.expval(qml.PauliZ(0)),
#             ]

#     # Select the appropriate circuit based on model type
#     if args.model == "amp_mean":
#         return classification_circuit_amp_mean(args, weights, features)
#     elif args.model == "pair_encoding":
#         return classification_circuit_different_combine_pairs(args, weights, features, trained_encoder)
#     else:
#         return classification_circuit(args, weights, features, trained_encoder)

# import pennylane as qml
# import torch
# import numpy as np
# from QEncoder_SP500_prediction.device_router import route_device
# from QEncoder_SP500_prediction.modules.layers import p, autoencoder_circuit_no_swap  # Absolute Import

# def keep_feature_from_padding(args, feature, time_step):
#     is_padded = np.all(np.isclose(feature, feature[0]))
#     pad_to = (args.version % 5) * 4
#     if is_padded and time_step > pad_to:
#         return False
#     return True

# def construct_classification_circuit(args, weights, features, trained_encoder=None):
#     sentiment_dev = route_device(args, 'medium')

#     if args.mode == 'train':
#         qnode = qml.QNode(lambda *qargs: insert_noise(qml.QNode(*qargs)), sentiment_dev, interface="torch", diff_method="backprop")
#     else:
#         qnode = qml.QNode(lambda *qargs: insert_noise(qml.QNode(*qargs)), sentiment_dev, interface="torch")

#     def classification_circuit(args, p_weights, features, trained_encoder=None):
#         num_latent, num_trash = args.num_latent, args.num_trash
#         if args.model not in ["ablation_angle", "ablation_angle_amp"]:
#             e_weights = torch.tensor(trained_encoder.weights.detach(), requires_grad=False)
#         p_weights = p_weights.reshape(args.sentence_len, -1)

#         initial_feature = features[0]

#         qml.AngleEmbedding(
#             initial_feature[: num_trash + num_latent],
#             wires=range(num_latent + num_trash),
#             rotation="X",
#         )
        
#         p(p_weights[0], args)

#         for i in range(1, len(features)):
#             if args.model == "ablation_angle":
#                 qml.AngleEmbedding(
#                     features[i][: args.num_trash], wires=range(num_trash), rotation="X"
#                 )
#                 qml.AngleEmbedding(
#                     features[i][num_trash : 2 * (num_trash)],
#                     wires=range(num_trash),
#                     rotation="Y",
#                 )
#                 p(p_weights[i], args)
#             elif args.model == "ablation_angle_amp":
#                 emb_wires = int(np.ceil(np.log2(2 * num_trash)))
#                 state = features[i][: (2 * num_trash)]
#                 state = state / np.linalg.norm(state)
#                 padded_state = np.zeros(2**emb_wires)
#                 padded_state[:state.shape[0]] = state
#                 if not np.isnan(state[0]):
#                     qml.MottonenStatePreparation(padded_state, wires=range(emb_wires))
#                 p(p_weights[i], args)
#             else:
#                 if args.pad_mode != 'selective' or keep_feature_from_padding(args, features[i], i):
#                     autoencoder_circuit_no_swap(e_weights, args)
#                     qml.AngleEmbedding(
#                         features[i][: args.num_trash], wires=range(num_trash), rotation="X"
#                     )
#                     qml.AngleEmbedding(
#                         features[i][num_trash : 2 * (num_trash)],
#                         wires=range(num_trash),
#                         rotation="Y",
#                     )
#                     p(p_weights[i], args)

#         p(p_weights[-1], args)

#         if args.mode == "train":
#             return qml.probs(0)
#         else:
#             return [
#                 qml.expval(qml.PauliX(0)),
#                 qml.expval(qml.PauliY(0)),
#                 qml.expval(qml.PauliZ(0)),
#             ]

#     classification_circuit = qnode(classification_circuit)

#     def classification_circuit_amp_mean(args, p_weights, features):
#         features = np.mean(features, axis=0)
#         p_weights = p_weights.reshape(3, -1)
#         qml.AmplitudeEmbedding(
#             features,
#             wires=range(args.num_latent + args.num_trash),
#             pad_with=0,
#             normalize=True,
#         )
#         for i in range(1):
#             p(p_weights[i], args)
#         if args.mode == "train":
#             return qml.probs(0)
#         else:
#             return [
#                 qml.expval(qml.PauliX(0)),
#                 qml.expval(qml.PauliY(0)),
#                 qml.expval(qml.PauliZ(0)),
#             ]

#     classification_circuit_amp_mean = qnode(classification_circuit_amp_mean)

#     def classification_circuit_different_combine_pairs(
#         args, p_weights, features, trained_encoder
#     ):
#         num_latent, num_trash = args.num_latent, args.num_trash
#         e_weights = torch.tensor(trained_encoder.weights.detach(), requires_grad=False)
#         p_weights = p_weights.reshape(args.sentence_len, -1)

#         initial_feature = features[0]
#         second_feature = features[1]

#         qml.AngleEmbedding(
#             initial_feature[: num_trash + num_latent],
#             wires=range(num_latent + num_trash),
#             rotation="X",
#         )
#         qml.AngleEmbedding(
#             second_feature[: num_trash + num_latent],
#             wires=range(num_latent + num_trash),
#             rotation="Y",
#         )
#         p(p_weights[0], args)
#         for i in range(2, len(features), 2):
#             autoencoder_circuit_no_swap(e_weights, args)
#             qml.AngleEmbedding(
#                 features[i][: args.num_trash], wires=range(num_trash), rotation="X"
#             )
#             if i != len(features) - 1:
#                 qml.AngleEmbedding(
#                     features[i + 1][: args.num_trash], wires=range(num_trash), rotation="Y"
#                 )
#             p(p_weights[i], args)
#         p(p_weights[-1], args)

#         if args.mode == "train":
#             return qml.probs(0)
#         else:
#             return [
#                 qml.expval(qml.PauliX(0)),
#                 qml.expval(qml.PauliY(0)),
#                 qml.expval(qml.PauliZ(0)),
#             ]

#     classification_circuit_different_combine_pairs = qnode(classification_circuit_different_combine_pairs)

#     if args.model == "amp_mean":
#         return classification_circuit_amp_mean(args, weights, features)
#     elif args.model == "pair_encoding":
#         return classification_circuit_different_combine_pairs(args, weights, features, trained_encoder)
#     else:
#         return classification_circuit(args, weights, features, trained_encoder)
