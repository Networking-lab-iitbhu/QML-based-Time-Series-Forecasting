import os
import torch
from .metrics import metrics

def test(
    model,
    args,
    test_dir,
    test_set,
    labels_test,
):
    print(test_dir)
    experiment = f"{args.dataset}_{args.loss}_{args.depth}_{args.n_cells}_{args.num_latent}_{args.num_trash}"
    checkpoint = torch.load(f"{test_dir}/{experiment}_weights_iteration_1",weights_only = False)
    state_dict = checkpoint["model_state_dict"]
    print(checkpoint['epoch'])
    if 'p_weights' in state_dict:
        state_dict['model_weights']=state_dict.pop('p_weights')
    model.load_state_dict(state_dict,strict=False)
    
    model.eval()
    
    with torch.no_grad():
        test_out = model(torch.tensor(test_set))
    
    metrics(test_out,labels_test)
    

    
    
    
    
    