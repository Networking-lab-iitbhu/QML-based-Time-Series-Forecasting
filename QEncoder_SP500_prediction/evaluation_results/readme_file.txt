In this folder we have:
-----------------------------------------------------------------------------------------------

accs - accuracies for each experiment conducted
file name is of the form:
experiment = f"{args.dataset}_{args.loss}_{args.depth}_{args.num_latent}_{args.num_trash}"
-> using the trained encoder we trained the model 10 times and then used the best weights each time 
   and predicted the outputs and plotted r2,mse,mae,mape

->test_results : predicted values on the test set for each time (out of 10)
weights -> best weights of encoder in each iteration. (we saved the model based on validation set losses)