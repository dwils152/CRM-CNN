import os
import sys
import numpy as np
from utils import split_data, split_kfold, get_data_loader
import utils
import torch
import torch.nn as nn
from ray import tune, air
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import CLIReporter
from ray.air.checkpoint import Checkpoint
from ray.tune.stopper import TrialPlateauStopper
from torchmetrics.classification import AUROC, FBetaScore
from model import CrmCNN
from sklearn import metrics
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt
import pickle
from torch.optim.lr_scheduler import StepLR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

def train(config, seqs, labels, checkpoint_dir, fasta, length):

    net = CrmCNN(config)
    device = "cpu"
    
    if torch.cuda.is_available():
        device = "cuda:0" 
    net.to(device)

    #pos_weights = torch.tensor([0.03]).to(device)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config["learning_rate"],
        #momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    #scheduler = StepLR(optimizer, step_size=0.1, gamma=0.1)

    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    trainset, valset, _ = split_data(seqs,
                                     labels,
                                     0.025, 0.025, 0.95,
                                     fasta,
                                     length,
                                     use_annotations=False)
    
    
    
    trainloader = utils.get_data_loader(trainset, config['batch_size'])
    valloader = utils.get_data_loader(valset, config['batch_size'])
    auroc_metric = AUROC(task="binary")  # Instantiate once
    fbeta_metric = FBetaScore(task="binary", beta=1.0).to(device)
    
    batch_gradient_norms = []
    batch_loss = []
    batch_correct = []

    for epoch in range(50):
        
        print("Training...")
        per_epoch_val_loss = 0
        per_epoch_val_correct = 0
        per_epoch_val_total = 0
        per_epoch_val_pred_list = []  # use list for collecting predictions
        per_epoch_val_labels_list = []

        net = net.train()

        # Training
        for batch_idx, (inputs, labels) in enumerate(trainloader, 1):
            percentage = (batch_idx / len(trainloader)) * 100
            if batch_idx % 100 == 0:
                print(f" Train Epoch: {epoch+1}, Batch %: {batch_idx}/{len(trainloader)} ({percentage:.2f}%)")

            # Input
            inputs = torch.unsqueeze(inputs, 1).float().to(device)
            labels = torch.squeeze(labels).float().to(device)
            predictions = net(inputs)
            predictions = torch.squeeze(predictions)

            #binary_pred, continuous_pred = net(inputs)
            #binary_loss = nn.BCELoss()(binary_pred, class_label)
            #continuous_loss = nn.MSELoss()(continuous_pred, score_label)  # or use MAE

            #binary_weight = 0.5  # Adjust based on importance or scale of losses
            #significance_weight = 0.5  # Adjust based on importance or scale of losses

            #loss = binary_weight * binary_loss + significance_weight * continuous_loss

            #print(predictions.shape)
            #print(labels.shape)
            
            

            
            # Round logits to get predictions
            rounded_predictions = torch.round(predictions)
            #rounded_predictions = (predictions >= 0.2).float()
            #num_correct = rounded_predictions.flatten().eq(labels.flatten()).sum().item()
            num_correct = rounded_predictions.eq(labels).sum().item()
            
            loss = criterion(predictions, labels)
            batch_loss.append(loss.item())
            #print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            
            # Printing the gradient norm
            #clip_grad_norm_(net.parameters(), 1.0)
            # Initialize an empty dictionary to store gradient norms for each layer
            gradient_norms = {name: [] for name, _ in net.named_parameters() if _.requires_grad}

            # When updating your model, after backward(), you can update the gradient_norms dictionary
            for name, param in net.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        # Append the current gradient norm to the list corresponding to this layer
                        gradient_norms[name].append(param.grad.norm().item())
                        
                        
            optimizer.step()
            #scheduler.step()

            # Plot the gradient norms
            for name, norms in gradient_norms.items():
                batch_gradient_norms.append(np.mean(norms))
                

            
            # Print the percentage of correct predictions and current loss
            if batch_idx % 100 == 0:
                print(f"Correct predictions:{num_correct/len(labels.flatten())*100:.2f}%")
                print(loss.item())
            
            # Move the data to the CPU to save GPU memory            
            predictions = predictions.detach().cpu() # tensor([batch x length])
            rounded_predictions = rounded_predictions.detach().cpu()
            labels = labels.detach().cpu() # tensor([batch x length])   
            num_correct = rounded_predictions.eq(labels).sum().item()
            batch_correct.append(num_correct/len(labels.flatten())*100)  # tensor([batch x length])

        net = net.eval()
        # Validation
        for batch_idx, (inputs, labels) in enumerate(valloader, 1):
            percentage = (batch_idx / len(valloader)) * 100
            if batch_idx % 100 == 0:
                print(f" Validation Epoch: {epoch+1}, Batch %: {batch_idx}/{len(valloader)} ({percentage:.2f}%)")
            
            # Inputs
            inputs = torch.unsqueeze(inputs, 1).float().to(device, non_blocking=True)
            labels = torch.squeeze(labels).float().to(device, non_blocking=True)
            #class_label, score_label = labels
            #class_label = torch.squeeze(class_label, 1).float().to(device)
            #score_label = torch.squeeze(score_label, 1).float().to(device)

            # Outputs
            with torch.no_grad():
                
                predictions = net(inputs)
                predictions = torch.squeeze(predictions)

                #binary_loss = nn.BCELoss()(binary_pred, class_label)
                #continuous_loss = nn.MSELoss()(continuous_pred, score_label)  # or use MAE

                loss = criterion(predictions, labels)

                #binary_weight = 0.5  # Adjust based on importance or scale of losses
                #significance_weight = 0.5  # Adjust based on importance or scale of losses

                #loss = binary_weight * binary_loss + significance_weight * continuous_loss
                
                # Move the data to the CPU to save GPU memory 
                predictions = predictions.detach().cpu()
                rounded_predictions = torch.round(predictions).detach().cpu()
                #rounded_predictions = (predictions >= 0.2).float()
                labels = labels.detach().cpu()
                num_correct = rounded_predictions.eq(labels).sum().item()
                
                per_epoch_val_total += len(labels.flatten())
                per_epoch_val_correct += num_correct
                per_epoch_val_loss += loss.item()
                per_epoch_val_pred_list.append(predictions.flatten())
                per_epoch_val_labels_list.append(labels.flatten())
                
        # Flatten the list of tensors into a single tensor
        per_epoch_val_pred_list = torch.cat(per_epoch_val_pred_list)
        per_epoch_val_labels_list = torch.cat(per_epoch_val_labels_list)

        # Checkpoint after each epoch
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save((net.state_dict(), optimizer.state_dict()), f"{checkpoint_dir}/checkpoint.pt")
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        
        # Report metrics
        auroc = auroc_metric(per_epoch_val_pred_list, per_epoch_val_labels_list)
        fbeta = fbeta_metric(per_epoch_val_pred_list, per_epoch_val_labels_list)
        session.report({"loss": per_epoch_val_loss/(epoch+1),
                        "fbeta": fbeta.item(),
                        "accuracy": per_epoch_val_correct/per_epoch_val_total,
                        "auroc": auroc.item()}, checkpoint=checkpoint)

    #plt.plot(batch_gradient_norms)
    #plt.xlabel('Batches')
    #plt.ylabel('Average Gradient Norm')
    #plt.title('Gradient Norm Evolution Over Batches')
    #plt.savefig('/users/dwils152/gradient_norm.png', dpi=300)
    #plt.clf()
    
    plt.plot(batch_loss)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.savefig('/users/dwils152/loss.png', dpi=300)
    plt.clf()
    
    plt.plot(batch_correct) 
    plt.xlabel('Batches')
    plt.ylabel('Accuracy %')
    plt.savefig('/users/dwils152/accuracy.png', dpi=300)
        
def test_best_model(best_result, seqs, labels,npy_dir, fasta, length):
    
    best_trained_model = CrmCNN(
        {
        "batch_size": best_result.config["batch_size"],
        "conv_stride": best_result.config["conv_stride"],
        "pool_stride": best_result.config["pool_stride"],
        "pool_size": best_result.config["pool_size"],
        "dropout": best_result.config["dropout"],
        "kernels": best_result.config["kernels"],
        "layer_n_kernels": best_result.config["layer_n_kernels"], 
        "kernel_len": best_result.config["kernel_len"],
        "learning_rate": best_result.config["learning_rate"],
        "weight_decay": best_result.config["weight_decay"],
        "momentum": best_result.config["momentum"],
        "dilation": best_result.config["dilation"],
        "num_layers": best_result.config["num_layers"],
        "act_fn": best_result.config["act_fn"],
        "training_data_len": int(length)   
        }
    )
    
    print(best_result.checkpoint)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(
        best_result.checkpoint.to_directory(), "checkpoint.pt")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    # Test the best model on the test set.
    _, _, testset = utils.split_data(seqs, labels,  0.6, 0.2, 0.2, fasta, length, use_annotations=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)

    annotations = []
    all_labels = []
    all_predictions = []
    all_unrounded_predictions = []
    total_correct = 0
    total = 0
    with torch.no_grad():
        best_trained_model = best_trained_model.eval()
        for data in testloader:
            #inputs, labels, annot = data
            #annotations.append(annot)
            inputs = torch.unsqueeze(inputs, 1).float()
            labels = labels.float()
            
            # Find indices of labels that are not 0 or 1
            #DEFAULT_VALUE = 0
            #labels[(labels != 0) & (labels != 1)] = DEFAULT_VALUE


            inputs, labels = inputs.to(device), labels.to(device)
            
            labels = labels.flatten().detach().cpu()
            predictions = best_trained_model(inputs).flatten().detach().cpu()
            rounded_predictions = torch.round(predictions).detach().cpu()
            num_correct = rounded_predictions.eq(labels).sum().item()
            total_correct += num_correct
            total += len(labels)

            all_labels.append(labels)
            all_predictions.append(rounded_predictions)
            all_unrounded_predictions.append(predictions)

    print("Test set accuracy: {}".format(total_correct/total))

    # Compute Metrics
    all_labels = torch.cat(all_labels).tolist()

    print(set(all_labels))
    
    all_labels = [round(i) for i in all_labels]
    
    print(set(all_labels))
    
    print(len(all_labels))
    print(len(all_predictions))
    
    #all_labels = [1 for i in all_labels if i >= 1]
    
    all_predictions = torch.cat(all_predictions).tolist()
    all_unrounded_predictions = torch.cat(all_unrounded_predictions).tolist()
    
    # AUC
    fpr, tpr, thresholds = metrics.roc_curve(
        all_labels, all_unrounded_predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(f"AUC: {auc}")
    
    # AP
    precision, recall, thresholds = metrics.precision_recall_curve(
         all_labels, all_unrounded_predictions, pos_label=1)
    ap = metrics.average_precision_score(all_labels, all_unrounded_predictions)
    
    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(all_labels, all_predictions)
    
    # Feature Vector
    feature_vector = best_trained_model.features
    os.makedirs(npy_dir, exist_ok=True)
    pickle.dump(feature_vector, open(f"{npy_dir}/feature_vector.pkl", "wb"))
    
    
    np.save(f"{npy_dir}/fpr.npy", fpr)
    np.save(f"{npy_dir}/tpr.npy", tpr)
    np.save(f"{npy_dir}/thresholds_auc.npy", thresholds)
    np.save(f"{npy_dir}/precision.npy", precision)
    np.save(f"{npy_dir}/recall.npy", recall)
    np.save(f"{npy_dir}/thresholds_precision_recall.npy", thresholds)
    np.save(f"{npy_dir}/confusion_matrix.npy", confusion_matrix)
    #np.save(f"{npy_dir}/feature_vector.npy", cat_feature_vector)

    with open(f"{npy_dir}/metrics.txt", "w") as fout:
        fout.write(f'Area Under ROC Curve: {auc}\n')
        fout.write(f'Precision: {precision}\n')
        fout.write(f'Recall: {recall}\n')
        fout.write(f'Average Precision: {ap}\n')
        
    kernels = best_trained_model.first_layer_kernels
    np.save(f"{npy_dir}/kernels.npy", kernels)
    #np.save(os.path.join(npy_dir, "labels.npy"), all_labels.numpy())


def main():
    #ray.inite
    organism = sys.argv[1] 
    length = sys.argv[2]

    project_dir = '/projects/zcsu_research1/dwils152/Motif-Cnn'

    if organism == 'mouse':
        seqs = f'{project_dir}/Data-v2/mouse_150_1e-6_rm-blacklist/SEQS_200.npy'
        labels = f'{project_dir}/Data-v2/mouse_150_1e-6_rm-blacklist/LABELS_200.npy'
        checkpoint_dir = f'{project_dir}/Results/{organism}_{length}/checkpoints'
        npy_dir = f'{project_dir}/Results/{organism}_{length}/npy_data'
        fasta = f'{project_dir}/Data-v2/mouse_150_1e-6_rm-blacklist/pos_neg_200.fa'
        
    
    config = {
        "batch_size": tune.choice([8, 16, 32, 64]),
        "conv_stride": tune.randint(1, 2),
        "pool_stride": tune.randint(1, 5),
        "pool_size": tune.randint(1, 5),
        "dropout": tune.quniform(0.05, 0.80, 0.05),
        "kernels": tune.randint(50, 201),
        "layer_n_kernels": tune.randint(25, 101),
        "kernel_len": tune.randint(7, 22),
        "learning_rate": tune.uniform(1e-6, 1e-3),
        "weight_decay": tune.uniform(1e-7, 1e-5),
        "momentum": tune.uniform(0.70, 0.99),
        "dilation": tune.choice([1]),
        "num_layers": tune.choice([1, 2, 3]),
        "act_fn": tune.choice(["leaky_relu"]),
        "training_data_len": int(length)
    }


    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=50,
        grace_period=1,
        reduction_factor=4,
        brackets=1
    )

    optuna_search = OptunaSearch(metric="fbeta", mode="max")
    stopper = TrialPlateauStopper(metric="loss", grace_period=1)
    reporter = CLIReporter()
    reporter.add_metric_column("auroc")
    reporter.add_metric_column("loss")
    reporter.add_metric_column("accuracy")
    reporter.add_metric_column("fbeta")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, seqs=seqs,
                                 labels=labels,
                                 checkpoint_dir=checkpoint_dir,
                                 fasta=fasta,
                                 length=int(length)),
            resources={"cpu": 32, "gpu": 2}
        ),
        param_space=config,
        tune_config=tune.TuneConfig(
            reuse_actors=False,
            max_concurrent_trials=1,
            search_alg=optuna_search,
            scheduler=asha_scheduler,
            num_samples=1,
            metric="loss",
            mode="min"
        ),
        run_config=air.RunConfig(
            name="bayes_opt_training",
            progress_reporter=reporter,
            stop=stopper,
            storage_path=f"/projects/zcsu_research1/dwils152/Motif-Cnn/Results/{organism}_{length}",
            sync_config=tune.SyncConfig(
                syncer=None
            )
        )
    )
    
    result = tuner.fit()
    best_trial = result.get_best_result("auroc", "max")
    print(best_trial)
    #test_best_model(best_trial, seqs, labels, npy_dir, fasta, int(length))


if __name__ == "__main__":
    print(os.cpu_count())
    main()
