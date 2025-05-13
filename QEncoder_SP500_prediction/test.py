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
    experiment = f"{args.dataset}_{args.loss}_{args.depth}_{args.n_cells}_{args.num_latent}_{args.num_trash}"
    checkpoint = torch.load(os.path.join(test_dir,f"{experiment}_weights"),weights_only = False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        test_out = model(torch.tensor(test_set, dtype=torch.float32))
    
    metrics(test_out,labels_test)
    

    
    
    
    
    