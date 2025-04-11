import torch
from pennylane import numpy as np
from tqdm import tqdm
import boto3
def evaluate(
    model,
    pca_glove_train,
    labels_train,
    pca_glove_val,
    labels_val,
    args,
):
    if args.aws == 'on':
        s3 = boto3.client('s3')
    if args.data_eval_cat == 'val':
        data = pca_glove_val
        labels = labels_val
    else:
        data = pca_glove_train
        labels = labels_train
    data = [i[: args.sentence_len] for i in data]
    model_outputs = []
    if args.mode == 'eval':
        for member in tqdm(range(1,args.n_members+1)):
            model.args.version = member
            weights_file = f'evaluation_results/weights/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.anz_set}_weights'
            model_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_{args.data_eval_cat}_{args.anz_set}_data'
            label_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_{args.data_eval_cat}_{args.anz_set}_labels'
            if args.aws == 'on':
                s3.download_file(args.s3_bucket, weights_file, weights_file)
            state_dict = torch.load(weights_file)['model_state_dict']
            model.load_state_dict(state_dict)
            outputs = [i.detach().numpy() for i in model(data)]
            model_outputs.append(outputs)
            print(model_output_file)
            np.save(model_output_file, model_outputs)
            np.save(label_output_file, labels)
            if args.aws == 'on':
                s3.upload_file(model_output_file +'.npy', args.s3_bucket, model_output_file+'.npy')
                s3.upload_file(label_output_file + '.npy', args.s3_bucket, label_output_file+'.npy')
    elif args.mode == 'eval_part':
        member = model.args.version
        weights_file = f'evaluation_results/weights/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.anz_set}_weights'
        model_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_{args.data_eval_cat}_{args.anz_set}_part_data'
        label_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_{args.data_eval_cat}_{args.anz_set}_part_labels'
        if args.aws == 'on':
            s3.download_file(args.s3_bucket, weights_file, weights_file)
        state_dict = torch.load(weights_file)['model_state_dict']
        model.load_state_dict(state_dict)
        outputs = [i.detach().numpy() for i in model(data)]
        model_outputs.append(outputs)
        print(model_output_file)
        np.save(model_output_file, model_outputs)
        np.save(label_output_file, labels)
        if args.aws == 'on':
            s3.upload_file(model_output_file +'.npy', args.s3_bucket, model_output_file+'.npy')
            s3.upload_file(label_output_file + '.npy', args.s3_bucket, label_output_file+'.npy')
