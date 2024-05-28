import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import random
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import argparse


plt.rc('legend', fontsize=30)
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', titlesize=24)
plt.rc('figure', titlesize=30)

plt.rcParams.update({
    'figure.constrained_layout.use': True,
    "pgf.texsystem": "xelatex",
    "font.family": "serif", 
    'pgf.rcfonts': False, # Disables font replacement
    "pgf.preamble": "\n".join([
        r'\usepackage{mathtools}'
        r'\usepackage{fontspec}'
        r'\usepackage[T1]{fontenc}'
        r'\usepackage{kpfonts}'
        r'\makeatletter'
        r'\AtBeginDocument{\global\dimen\footins=\textheight}'
        r'\makeatother'
    ]),
})


cwd = os.getcwd()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
use_cuda = torch.cuda.is_available()


experiment1 = ['DTU_gen_vs_AAU_gen_subset','DTU_gen_vs_AAU_gen_subset_test']
experiment2 = ['DTU_gen_vs_AAU_gen_v1', 'DTU_vs_AAU_test']
experiment3 = ['DTU_gen_vs_AAU_gen_v1', 'DTU_vs_lfw_test']
experiment4 = ['DTU_gen_vs_AAU_gen_v1', 'DTU_seen_vs_DTU_unseen_test']
experiment5 = ['DTU_wm_vs_AAU', 'DTU_wm_vs_AAU_unseen_test']
experiment6 = ['DTU_hwm_vs_AAU', 'DTU_hwm_vs_AAU_unseen_test']
experiment7 = ['DTU_gen_vs_AAU_gen_v1', 'DTU_vs_AAU_unseen_test']

experiment_set = [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7]
#experiment_set = [experiment5, experiment6]
#experiment_set = [experiment1, experiment2, experiment3, experiment4]

num_experiments = len(experiment_set)

root_im = cwd + os.sep + 'images_attack_model' + os.sep 

root_data = []
root_data_test = []
for experiment in experiment_set:
    root_data.append(root_im + experiment[0] + os.sep)
    root_data_test.append(root_im + experiment[1] + os.sep)

REPEAT_EXPERIMENT = 5


def create_dataset(parent_dir):
    # Define your transformations
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(parent_dir, transform=transform)
    
    return dataset


def train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, device):
    model.to(device)
    start_time = time.time()
    train_loss = []
    train_accuary = []
    val_accuary = []
    val_loss = []
    for epoch in range(num_epochs): #(loop for every epoch)
        print("Epoch {} running".format(epoch)) #(printing message)
        """ Training Phase """
        model.train()    #(training model)
        running_loss = 0.   #(set loss 0)
        running_corrects = 0 
        # load a batch data of images
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            # forward inputs and get output
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # get loss value and update the network weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset) * 100
        # Append result
        train_loss.append(epoch_loss)
        train_accuary.append(epoch_acc)
        # Print progress
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))
        
        """ Validation Phase """
        model.eval()    #(evaluation model)
        running_loss = 0.   #(set loss 0)
        running_corrects = 0 
        # load a batch data of images
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            # forward inputs and get output
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects / len(val_loader.dataset) * 100
        # Append result
        val_loss.append(epoch_loss)
        val_accuary.append(epoch_acc)
        # Print progress
        print('[Validation #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))

    return model, train_loss, train_accuary, val_loss, val_accuary

def test_model(model, test_loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            # Get the probability of the positive class
            y_pred += probs[:, 1].tolist()
            y_true += labels.tolist()

    return y_pred, y_true

def run_experiment(root_data, root_data_test, experiment_set):
    for i in range(num_experiments):
        print(f'Experiment {i+1} of {len(experiment_set)}')
        roc_auc = []
        tpr = []
        fpr = []
        thresholds = []
        y_pred = []
        y_true = []

        for seed in range(REPEAT_EXPERIMENT):
            print(f'Seed {seed+1} of {REPEAT_EXPERIMENT}')
            print(f'Loading data')

            # Set seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

            # Load data
            train_data = create_dataset(root_data[i])
            test_data = create_dataset(root_data_test[i])
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
            try:
                test_data, val_set = torch.utils.data.random_split(test_data, [int(0.85*len(test_data)), int(0.15*len(test_data))])
            except:
                test_data, val_set = torch.utils.data.random_split(test_data, [int(0.85*len(test_data))+1, int(0.15*len(test_data))])
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

            # Define the model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)

            # Define the loss function
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train the model
            print('Training model')
            num_epochs = 20

            # if there already exists a model, load it for the current seed
            if os.path.exists(f'{root_data[i]}model_{seed}.pt'):
                model.load_state_dict(torch.load(f'{root_data[i]}model_{seed}.pt'))
                model.to(device)
                model.eval()
            
            else:
                model, train_loss, train_accuary, val_loss, val_accuary = train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, device)

                # Save the model
                torch.save(model.state_dict(), f'{root_data[i]}model_{seed}.pt')

                # Save the train loss and train accuracy as a matplotlib plot to the training folder
                
                plt.size = (8, 6)
                plt.plot(train_loss)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Train Loss')
                plt.savefig(f'{root_data[i]}train_loss_{seed}_{i}.png')
                # Save with pgf
                plt.savefig(f'{root_data[i]}train_loss_{seed}_{i}.pgf')
                plt.close()

                plt.plot(train_accuary)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Train Accuracy')
                plt.savefig(f'{root_data[i]}train_accuracy_{seed}_{i}.png')
                # Save the pgf code to a file
                plt.savefig(f'{root_data[i]}train_accuracy_{seed}_{i}.pgf')
                plt.close()

                plt.plot(val_loss)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Validation Loss')
                plt.savefig(f'{root_data[i]}validation_loss_{seed}_{i}.png')
                # Save the pgf code to a file
                plt.savefig(f'{root_data[i]}validation_loss_{seed}_{i}.pgf')
                plt.close()
                
                plt.plot(val_accuary)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Validation Accuracy')
                plt.savefig(f'{root_data[i]}validation_accuracy_{seed}_{i}.png')
                # Save the tikz code to a file
                plt.savefig(f'{root_data[i]}validation_accuracy_{seed}_{i}.pgf')
                plt.close()

            # Test the model, we repeat this step for each seed and save the results
            print('Testing model')
            y_pred_, y_true_ = test_model(model, test_loader)
            fpr_, tpr_, thresholds_ = roc_curve(y_true_, y_pred_)
            roc_auc_ = roc_auc_score(y_true_, y_pred_)
            roc_auc.append(roc_auc_)
            tpr = tpr + [tpr_]
            fpr = fpr + [fpr_]
            thresholds.append(thresholds_)
            y_pred.append(y_pred_)
            y_true.append(y_true_)


        # Make a plot which shows the confidence interval of the ROC curve

        plt.figure(figsize=(8, 6))
        # The tpr, fpr and thresholds are different sizes, so we need to interpolate them to the same size
        mean_tpr = np.mean([np.interp(np.linspace(0, 1, 100), fpr[i], tpr[i]) for i in range(REPEAT_EXPERIMENT)], axis=0)
        mean_fpr = np.mean([np.linspace(0, 1, 100) for i in range(REPEAT_EXPERIMENT)], axis=0)
        mean_auc = np.mean(roc_auc)
        sd_auc = np.std(roc_auc)
        sd_tpr = np.std([np.interp(np.linspace(0, 1, 100), fpr[i], tpr[i]) for i in range(REPEAT_EXPERIMENT)], axis=0)
        interval = 1.96 * (sd_auc / np.sqrt(REPEAT_EXPERIMENT))
        interval_tpr = 1.96 * (sd_tpr / np.sqrt(REPEAT_EXPERIMENT))
        tprs_upper = np.minimum(mean_tpr + interval_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - interval_tpr, 0)
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'ROC AUC = {mean_auc:.2f} $\pm$ {interval:.2f}')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'{root_data_test[i]}{experiment_set[i][1]}_ROC.png')
        plt.savefig(f'{root_im}figures/{experiment_set[i][1]}_ROC.pgf')
        plt.close()

        # Show the confusion matrix, which uses the mean of the ROC AUC as the threshold and has the confidence interval
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        cm = np.zeros((2, 2), dtype=object)
        
        # Count the number of true positives, true negatives, false positives and false negatives for each seed
        optimal_idx = [np.argmax(tpr[i] - fpr[i]) for i in range(REPEAT_EXPERIMENT)]
        optimal_threshold = np.array([thresholds[i][optimal_idx[i]] for i in range(REPEAT_EXPERIMENT)])

        true_positives = np.sum((y_pred >= optimal_threshold.reshape(-1, 1)) & (y_true == 1), axis=1)
        true_negatives = np.sum((y_pred < optimal_threshold.reshape(-1, 1)) & (y_true == 0), axis=1)
        false_positives = np.sum((y_pred >= optimal_threshold.reshape(-1, 1)) & (y_true == 0), axis=1)
        false_negatives = np.sum((y_pred < optimal_threshold.reshape(-1, 1)) & (y_true == 1), axis=1)

        # Calculate the mean and confidence interval 
        cm[0, 0] = f'{np.mean(true_negatives):.2f} \n [ {np.mean(true_negatives) - 1.96*np.std(true_negatives)/np.sqrt(REPEAT_EXPERIMENT):.2f}, {np.mean(true_negatives) + 1.96*np.std(true_negatives)/np.sqrt(REPEAT_EXPERIMENT):.2f} ]'
        cm[0, 1] = f'{np.mean(false_positives):.2f} \n [ {np.mean(false_positives) - 1.96*np.std(false_positives)/np.sqrt(REPEAT_EXPERIMENT):.2f}, {np.mean(false_positives) + 1.96*np.std(false_positives)/np.sqrt(REPEAT_EXPERIMENT):.2f} ]'
        cm[1, 0] = f'{np.mean(false_negatives):.2f} \n [ {np.mean(false_negatives) - 1.96*np.std(false_negatives)/np.sqrt(REPEAT_EXPERIMENT):.2f}, {np.mean(false_negatives) + 1.96*np.std(false_negatives)/np.sqrt(REPEAT_EXPERIMENT):.2f} ]'
        cm[1, 1] = f'{np.mean(true_positives):.2f} \n [ {np.mean(true_positives) - 1.96*np.std(true_positives)/np.sqrt(REPEAT_EXPERIMENT):.2f}, {np.mean(true_positives) + 1.96*np.std(true_positives)/np.sqrt(REPEAT_EXPERIMENT):.2f} ]'
        
        cm_numbers = np.array([[np.mean(true_negatives), np.mean(false_positives)], [np.mean(false_negatives), np.mean(true_positives)]])

        # Create a confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(data=cm_numbers, annot=cm, fmt='', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f'{root_data_test[i]}{experiment_set[i][1]}_CM.png')
        plt.savefig(f'{root_im}figures/{experiment_set[i][1]}_CM.pgf')
        plt.close()

        # Save the results to a pickle file
        results = {'roc_auc': roc_auc, 'tpr': tpr, 'fpr': fpr, 'thresholds': thresholds, 'y_pred': y_pred, 'y_true': y_true, 'cm': cm}
        pd.to_pickle(results, f'{root_data_test[i]}results.pkl')

        
    
# Function which loads the results and creates the plots
def create_plots(root_data_test, experiment_set):
    for i in range(len(experiment_set)):
        results = pd.read_pickle(f'{root_data_test[i]}results.pkl')
        roc_auc = results['roc_auc']
        tpr = results['tpr']
        fpr = results['fpr']
        thresholds = results['thresholds']
        y_pred = results['y_pred']
        y_true = results['y_true']
        cm = results['cm']

        # Make a plot which shows the confidence interval of the ROC curve
        plt.figure(figsize=(8, 6))
        # The tpr, fpr and thresholds are different sizes, so we need to interpolate them to the same size
        mean_tpr = np.mean([np.interp(np.linspace(0, 1, 100), fpr[i], tpr[i]) for i in range(REPEAT_EXPERIMENT)], axis=0)
        mean_fpr = np.mean([np.linspace(0, 1, 100) for i in range(REPEAT_EXPERIMENT)], axis=0)
        mean_auc = np.mean(roc_auc)
        sd_auc = np.std(roc_auc)
        sd_tpr = np.std([np.interp(np.linspace(0, 1, 100), fpr[i], tpr[i]) for i in range(REPEAT_EXPERIMENT)], axis=0)
        interval = 1.96 * (sd_auc / np.sqrt(REPEAT_EXPERIMENT))
        interval_tpr = 1.96 * (sd_tpr / np.sqrt(REPEAT_EXPERIMENT))
        tprs_upper = np.minimum(mean_tpr + interval_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - interval_tpr, 0)
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'ROC AUC = {mean_auc:.2f} $\pm$ {interval:.2f}')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'{root_data_test[i]}{experiment_set[i][1]}_ROC.png')
        plt.savefig(f'{root_im}figures/{experiment_set[i][1]}_ROC.pgf')
        plt.close()

        # Show the confusion matrix, which uses the mean of the ROC AUC as the threshold and has the confidence interval
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        cm = np.zeros((2, 2), dtype=object)

        # Count the number of true positives, true negatives, false positives and false negatives for each seed
        optimal_idx = [np.argmax(tpr[i] - fpr[i]) for i in range(REPEAT_EXPERIMENT)]
        optimal_threshold = np.array([thresholds[i][optimal_idx[i]] for i in range(REPEAT_EXPERIMENT)])

        true_positives = np.sum((y_pred >= optimal_threshold.reshape(-1, 1)) & (y_true == 1), axis=1)
        true_negatives = np.sum((y_pred < optimal_threshold.reshape(-1, 1)) & (y_true == 0), axis=1)
        false_positives = np.sum((y_pred >= optimal_threshold.reshape(-1, 1)) & (y_true == 0), axis=1)
        false_negatives = np.sum((y_pred < optimal_threshold.reshape(-1, 1)) & (y_true == 1), axis=1)

        # Calculate the mean and confidence interval
        cm[0, 0] = f'{np.mean(true_negatives):.2f} \n [ {np.mean(true_negatives) - 1.96*np.std(true_negatives)/np.sqrt(REPEAT_EXPERIMENT):.2f}, {np.mean(true_negatives) + 1.96*np.std(true_negatives)/np.sqrt(REPEAT_EXPERIMENT):.2f} ]'
        cm[0, 1] = f'{np.mean(false_positives):.2f} \n [ {np.mean(false_positives) - 1.96*np.std(false_positives)/np.sqrt(REPEAT_EXPERIMENT):.2f}, {np.mean(false_positives) + 1.96*np.std(false_positives)/np.sqrt(REPEAT_EXPERIMENT):.2f} ]'
        cm[1, 0] = f'{np.mean(false_negatives):.2f} \n [ {np.mean(false_negatives) - 1.96*np.std(false_negatives)/np.sqrt(REPEAT_EXPERIMENT):.2f}, {np.mean(false_negatives) + 1.96*np.std(false_negatives)/np.sqrt(REPEAT_EXPERIMENT):.2f} ]'
        cm[1, 1] = f'{np.mean(true_positives):.2f} \n [ {np.mean(true_positives) - 1.96*np.std(true_positives)/np.sqrt(REPEAT_EXPERIMENT):.2f}, {np.mean(true_positives) + 1.96*np.std(true_positives)/np.sqrt(REPEAT_EXPERIMENT):.2f} ]'
        
        cm_numbers = np.array([[np.mean(true_negatives), np.mean(false_positives)], [np.mean(false_negatives), np.mean(true_positives)]])

        # Create a confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(data=cm_numbers, annot=cm, fmt='', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f'{root_data_test[i]}{experiment_set[i][1]}_CM.png')
        plt.savefig(f'{root_im}figures/{experiment_set[i][1]}_CM.pgf')
        plt.close()



parser = argparse.ArgumentParser(description='Run the experiment')
parser.add_argument('--plot', action='store_true', help='Only create the plots')
parser.add_argument('--run_all', action='store_true', help='Run all experiments')
args = parser.parse_args()


if args.plot:
    create_plots(root_data_test, experiment_set)
    exit()

if args.run_all:
    run_experiment(root_data, root_data_test, experiment_set)
    exit()

