import torch
from pennylane import numpy as np
from models.stacking import StackingModel, accuracy
import torch.optim as optim
import boto3


def join_parts(args, cat):
    s3 = boto3.client('s3')
    all_predictions = []
    for member in range(1,6):
        file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_{cat}_{9}_part_data.npy'
        s3.download_file(args.s3_bucket,file, file)
        all_predictions.append(np.load(file)[0])
    return all_predictions

def aggregate_outputs(args):
    member = args.n_members
    model_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{args.anz_set}_data.npy'
    label_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{args.anz_set}_labels.npy'
    print(model_output_file)
    if args.aws == 'on':
        s3 = boto3.client('s3')
    if args.mode == 'majrule':
        if args.aws == 'on':
            s3.download_file(args.s3_bucket,model_output_file, model_output_file)
            s3.download_file(args.s3_bucket,label_output_file, label_output_file)
        predictions = list(np.load(model_output_file))
        labels_val = np.load(label_output_file)
        n_samples = predictions[0].shape[0]
        correct = 0
        for i in range(n_samples):
            guess = 0
            for mem in range(args.n_members):
                if predictions[mem][i][2] < 0:
                    guess -=1
                else:
                    guess +=1
            if labels_val[i] == 0 and guess < 0:
                correct+=1
            elif labels_val[i] == 1 and guess > 0:
                correct+=1
        print(f'Accuracy: {100*round(correct/n_samples, 2)}')
    if args.mode == 'single_member':
        if args.aws == 'on':
            if args.model != 'machine_aware':
                model_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{1}_{args.machine}_val_{args.anz_set}_data.npy'
                label_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{1}_{args.machine}_val_{args.anz_set}_labels.npy'
            else:
                model_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{15}_{args.machine}_val_{args.anz_set}_data.npy'
                label_output_file = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{15}_{args.machine}_val_{args.anz_set}_labels.npy'
            # model_output_file = f'evaluation_results/predictions/val_{args.dataset}_on_real_cairo_data.npy'
            # label_output_file = f'evaluation_results/predictions/val_{args.dataset}_on_real_cairo_labels.npy'
            s3.download_file(args.s3_bucket,model_output_file, model_output_file)
            s3.download_file(args.s3_bucket,label_output_file, label_output_file)
        predictions = list(np.load(model_output_file))
        labels_val = np.load(label_output_file)
        
        n_samples = predictions[0].shape[0]
        accs = []
        #for mem in range(args.n_members):
        if args.model != 'machine_aware':
            for mem in range(1):
                correct = 0
                for i in range(n_samples):
                    #print(predictions[mem][i][2], labels_val, i)
                    if predictions[mem][i][2] < 0 and labels_val[i] == 0:
                        correct +=1
                    elif predictions[mem][i][2] > 0 and labels_val[i] == 1:
                        correct +=1
                accs.append(correct/n_samples)
        else:
            mem = 4 # no padding
            correct = 0
            for i in range(n_samples):
                if predictions[mem][i][2] < 0 and labels_val[i] == 0:
                    correct +=1
                elif predictions[mem][i][2] > 0 and labels_val[i] == 1:
                    correct +=1
            accs.append(correct/n_samples)
        print(f'Accuracy: {100*round(np.mean(accs), 2)}')

    if 'stacking' in args.mode:
        if args.train_machine != 'None':
            model_output_file_train = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{args.anz_set}_data.npy'
            label_output_file_train = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{args.anz_set}_labels.npy'
        else:
            model_output_file_train = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{args.anz_set}_data.npy'
            label_output_file_train = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{args.anz_set}_labels.npy'           
        if args.aws == 'on':
            s3.download_file(args.s3_bucket,model_output_file, model_output_file)
            s3.download_file(args.s3_bucket,label_output_file, label_output_file)
            s3.download_file(args.s3_bucket,model_output_file_train, model_output_file_train)
            s3.download_file(args.s3_bucket,label_output_file_train, label_output_file_train)
                                           
        train_data = np.load(model_output_file_train)[:args.n_members]
        val_data = np.load(model_output_file)[:args.n_members]
        train_labels = np.load(label_output_file_train)
        val_labels = np.load(label_output_file)
        # if args.train_machine != 'None':
        #     noisy_train_data = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_data.npy'
        #     noisy_train_labels = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_labels.npy'
        # else:
        #     noisy_train_data = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_data.npy'
        #     noisy_train_labels = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_labels.npy'
        #     if args.aws == 'on':
        #         s3.download_file(args.s3_bucket,noisy_train_data, noisy_train_data)
        #         s3.download_file(args.s3_bucket,noisy_train_labels, noisy_train_labels)
        #     train_data = np.load(noisy_train_data)[:args.n_members]
        #     train_labels =np.load(noisy_train_labels)
        train_data = torch.swapaxes(torch.tensor(train_data), 0, 1)
        val_data = torch.swapaxes(torch.tensor(val_data), 0, 1)
        if args.mode == 'stacking_naive':
            train_data = train_data[:, :, 2]
            val_data = val_data[:, :, 2]
        batch_size =  val_data.shape[0]
        val_data = val_data.reshape([batch_size, -1])
        accs = []

        for _ in range(10):
            model = StackingModel(args)
            opt = optim.Adam(model.parameters(), lr=.01)
            loss_function = torch.nn.BCELoss()
            for i in range(10000):
                model.zero_grad()
                train_indecies= np.random.randint(0, len(train_data), (batch_size,))
                train_labels_batch = torch.tensor([int(train_labels[x]) for x in train_indecies]).to(torch.float32).reshape([batch_size, 1])
                
                features = torch.tensor(train_data[torch.tensor(train_indecies)]).to(torch.float32)
                features=features.reshape([batch_size, -1])
                out = model(features)
                loss = loss_function(out, train_labels_batch)
                loss.backward()
                opt.step()
            out = model(val_data).reshape([-1])
            accs.append(accuracy(val_labels, out))
        print(f'Accuracy: {100*round(np.mean(accs), 2)}')
    if args.mode == 'combine' or args.mode == 'combine_naive':
        # 3-> E
        # 4-> A
        # 5-> B
        # 6-> C
        # 7-> D
        if args.train_machine != 'None':
            #model_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{4}_data.npy'
            model_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{4}_data.npy'

            model_output_file_train2 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{2}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{5}_data.npy'
            model_output_file_train3 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{1}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{6}_data.npy'

            label_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{4}_labels.npy'
            
        else:
            model_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{4}_data.npy'
            model_output_file_train2 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{2}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{5}_data.npy'
            model_output_file_train3 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{1}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{6}_data.npy'
            label_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{4}_labels.npy'   

        model_output_file1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{4}_data.npy'
        model_output_file2 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{2}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{5}_data.npy'
        model_output_file3 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{1}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{6}_data.npy'
        # model_output_file4 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{6}_data.npy'
        # model_output_file5 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{7}_data.npy'
        
        label_output_file1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{4}_labels.npy'

        s3.download_file(args.s3_bucket,model_output_file1, model_output_file1)
        s3.download_file(args.s3_bucket,model_output_file2, model_output_file2)
        s3.download_file(args.s3_bucket,model_output_file3, model_output_file3)
        s3.download_file(args.s3_bucket,model_output_file_train1, model_output_file_train1)
        s3.download_file(args.s3_bucket,model_output_file_train2, model_output_file_train2)
        s3.download_file(args.s3_bucket,model_output_file_train3, model_output_file_train3)
        s3.download_file(args.s3_bucket,label_output_file_train1, label_output_file_train1)
        s3.download_file(args.s3_bucket,label_output_file1, label_output_file1)
        train_data1 = join_parts(args, 'train')                          
        #train_data1 = np.load(model_output_file_train1)[:5]
        #val_data1 = np.load(model_output_file1)[:5]
        val_data1 = join_parts(args, 'val') 
        train_data2 = np.load(model_output_file_train2)[:5]
        val_data2 = np.load(model_output_file2)[:5]

        train_data3 = np.load(model_output_file_train3)[:5]
        val_data3 = np.load(model_output_file3)[:5]



        train_data = np.concatenate((train_data1, train_data2, train_data3))
        val_data = np.concatenate((val_data1, val_data2, val_data3))

        train_labels = np.load(label_output_file_train1)
        val_labels = np.load(label_output_file1)

        train_data = torch.swapaxes(torch.tensor(train_data), 0, 1)
        val_data = torch.swapaxes(torch.tensor(val_data), 0, 1)
        if args.mode ==  'combine_naive':
            train_data = train_data[:, :, 2]
            val_data = val_data[:, :, 2]
        batch_size =  val_data.shape[0]
        val_data = val_data.reshape([batch_size, -1])


        accs = []

        for _ in range(10):
            model = StackingModel(args)
            opt = optim.Adam(model.parameters(), lr=.01)
            loss_function = torch.nn.BCELoss()
            for i in range(10000):
                model.zero_grad()
                train_indecies= np.random.randint(0, len(train_data), (batch_size,))
                train_labels_batch = torch.tensor([int(train_labels[x]) for x in train_indecies]).to(torch.float32).reshape([batch_size, 1])
                
                features = torch.tensor(train_data[torch.tensor(train_indecies)]).to(torch.float32)
                features=features.reshape([batch_size, -1])
                out = model(features)
                loss = loss_function(out, train_labels_batch)
                loss.backward()
                opt.step()
            out = model(val_data).reshape([-1])
            accs.append(accuracy(val_labels, out))
        print(f'Accuracy: {100*round(np.mean(accs), 3)}')




    elif args.mode == 'majrule_combine':
        if args.train_machine != 'None':
            model_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{4}_data.npy'
            model_output_file_train2 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{2}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{5}_data.npy'
            model_output_file_train3 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{1}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{6}_data.npy'

            label_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.train_machine}_train_{4}_labels.npy'
            
        else:
            model_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{4}_data.npy'
            model_output_file_train2 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{2}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{5}_data.npy'
            model_output_file_train3 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{1}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{6}_data.npy'
            label_output_file_train1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_local_train_{4}_labels.npy'   

        model_output_file1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{3}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{4}_data.npy'
        model_output_file2 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{2}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{5}_data.npy'
        model_output_file3 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{1}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{6}_data.npy'
        # model_output_file4 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{6}_data.npy'
        # model_output_file5 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{7}_data.npy'
        
        label_output_file1 = f'evaluation_results/predictions/{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{member}_{args.machine}_val_{3}_labels.npy'

        s3.download_file(args.s3_bucket,model_output_file1, model_output_file1)
        s3.download_file(args.s3_bucket,model_output_file2, model_output_file2)
        s3.download_file(args.s3_bucket,model_output_file3, model_output_file3)
        s3.download_file(args.s3_bucket,model_output_file_train1, model_output_file_train1)
        s3.download_file(args.s3_bucket,model_output_file_train2, model_output_file_train2)
        s3.download_file(args.s3_bucket,model_output_file_train3, model_output_file_train3)
                                           
        train_data1 = np.load(model_output_file_train1)[:5]
        val_data1 = np.load(model_output_file1)[:5]
        train_data1 = join_parts(args, 'train')                          
        #train_data1 = np.load(model_output_file_train1)[:5]
        #val_data1 = np.load(model_output_file1)[:5]
        val_data1 = join_parts(args, 'val') 

        train_data2 = np.load(model_output_file_train2)[:5]
        val_data2 = np.load(model_output_file2)[:5]

        train_data3 = np.load(model_output_file_train3)[:5]
        val_data3 = np.load(model_output_file3)[:5]

        val_labels = np.load(label_output_file1)
        val_data = np.concatenate((val_data1, val_data2, val_data3))

        predictions = val_data
        labels_val = np.load(label_output_file1)
        n_samples = predictions.shape[1]
        correct = 0
        for i in range(n_samples):
            guess = 0
            for mem in range(15):
                #print(predictions[mem][i][2], guess, labels_val[i])
                if predictions[mem][i][2] < 0:
                    guess -=1
                else:
                    guess +=1
            if labels_val[i] == 0 and guess < 0:
                correct+=1
            elif labels_val[i] == 1 and guess > 0:
                correct+=1
        print(f'Accuracy: {100*round(correct/n_samples, 2)}')