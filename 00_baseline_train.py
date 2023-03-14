import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spread_train import *
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import torchmetrics


############################## CARGA DE DATOS ####################################

with open('dict_data_final.pkl', 'rb') as f:
    dict_data_final = pickle.load(f)

# Remove Beniarda and Bolulla in our dict_data_final

dict_data_final.pop('Beniarda')
dict_data_final.pop('Bolulla')

# set parameters

part = [0.1, 0.5, 0.9]
inc = 1
p0 = 0.25
div = 2
n_it = 10**2

# set metrics

confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=3)

def save_metrics(name, y_pred, y_true):

    
    dict_df = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(dict_df).transpose()
    df.to_csv('metrics/' + 'metrics_' + name + '.csv', index=False)

def fig_majority(name, y_true, y_vote):

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(y_true)
    ax[0].set_title('Real Burned Area')
    
    ax[1].imshow(y_vote)
    ax[1].set_title('Estimated Burned Area')

    fig.suptitle('Comparison of Real and Estimated Burned Areas (' + name + ')')

    plt.savefig('figs/png/' + 'fig_' + name + '.png')
    plt.savefig('figs/eps/' + 'fig_' + name + '.eps')


for name in dict_data_final.keys():

    print('Processing ' + name + '...')

    # Load data
    x, y = dict_data_final[name]

    # create grid 

    grid = Grid(x=x, y=y)
    grid.initialize(part=part, inc=inc, p0=p0, div=div)
    grid.submatrix()
    grid.enlargement_process()
    grid.montecarlo(n_it=n_it)

    y_true = y[:, :, -1].flatten()
    y_pred = grid.X[:, :, -1].numpy().flatten()
    save_metrics(name, y_pred, y_true)

    y_true = y[:, :, -1]
    y_pred = grid.X[:, :, -1].numpy()
    fig_majority(name, y_true, y_pred)

    confmat.update(grid.X[:, :, -1].flatten(), grid.y[:, :, -1].flatten())
    cm = confmat.compute()

    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('metrics/conf_matrix/' + 'confmat_' + name + '.png')
    np.save('metrics/conf_matrix/' + 'confmat_' + name + '.npy', cm.numpy())

    confmat.reset()

    # rate of expansion figures
    y0, y1, y2 = (grid.y==0).type(torch.int), (grid.y==1).type(torch.int), (grid.y==2).type(torch.int)

    # get sus_k such that sus_k = y0[:, :, k].sum()
    # get inf_k such that inf_k = y1[:, :, k].sum()
    # get dead_k such that dead_k = y2[:, :, k].sum()ç

    sus_k = y0.sum(dim=(0, 1))
    inf_k = y1.sum(dim=(0, 1))
    dead_k = y2.sum(dim=(0, 1))


    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    abscisas = np.argwhere(x.Train.values == True).flatten()

    ax[0].plot(grid.df_MC.Susceptible.values, '--', label='estimated')
    ax[0].plot(abscisas, sus_k.numpy(), label='real')
    ax[0].legend()
    
    ax[1].plot(grid.df_MC.Infected.values, '--', label='est inf')
    ax[1].plot(grid.df_MC.Dead.values, '--', label='est dead')
    ax[1].plot(abscisas, inf_k.numpy(), label='real inf')
    ax[1].plot(abscisas, dead_k.numpy(), label='real dead')
    ax[1].legend()

    fig.suptitle('Rate of expansion of the fire (' + name + ')')
    plt.savefig('figs/png/' + 'fig_rate_' + name + '.png')
    plt.savefig('figs/eps/' + 'fig_rate_' + name + '.eps')