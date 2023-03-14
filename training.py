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
import time

############################## CARGA DE DATOS ####################################

with open('dict_data_final.pkl', 'rb') as f:
    dict_data_final = pickle.load(f)

particiones = [[0.1, 0.5, 0.9], [0.05, 0.1, 0.3], [0.25, 0.5, 0.75], [0.1, 0.3, 0.6], [0, 0, 0.1]]
deltas = [1, 2, 4, 6]

###################################################################################

def get_probs(cl0, cl1, cl2):

    cl0 = cl0.flatten()
    cl1 = cl1.flatten()
    cl2 = cl2.flatten()

    probs = torch.stack([cl0, cl1, cl2], dim=1).reshape(-1, 3)

    return probs

# Función de pérdida

crit_class = nn.CrossEntropyLoss(reduction='sum')
crit_num = nn.L1Loss(reduction='sum')

losses = np.zeros(shape=(len(deltas), len(particiones)), dtype=np.float32)

c_x = 0
for inc in deltas:
    c_y = 0
    for part in particiones:
        loss = 0
        for name in dict_data_final.keys():

            print(part, name)

            x, y = dict_data_final[name]
            grid = Grid(x=x, y=y)
            grid.initialize(part=part, inc=inc)
            grid.submatrix()
            grid.enlargement_process()
            grid.montecarlo(n_it=25)
            
            try:
                grid.Train = torch.cat((torch.tensor([False]), grid.Train.type(torch.bool)), 0)
                probs = get_probs(grid.X0[:, :, grid.Train], grid.X1[:, :, grid.Train], grid.X2[:, :, grid.Train])
                y0 = (grid.y == 0).type(torch.float)
                loss += crit_class(probs, grid.y.flatten().long()) + crit_num(grid.X0[:, :, grid.Train], y0)
            except:

                print('No hay datos de entrenamiento en: ', name)
                pass
            
            losses[c_x, c_y] = loss
        c_y += 1
    c_x += 1

id_x, id_y = np.where(losses == np.min(losses))

inc = deltas[id_x]
part = particiones[id_y]

print('Best delta is: ', inc, ' and best partition is: ', part)

plt.plot(losses.flatten())
plt.show()

# save losses.npy and inc and part

np.save('losses.npy', losses)
np.save('inc.npy', np.array(inc))
np.save('part.npy', np.array(part))


####################### PARÁMETROS ############################

alpha = torch.tensor(2., requires_grad=True, dtype=torch.float)
beta = torch.tensor(2., requires_grad=True, dtype=torch.float)
gamma = torch.tensor(2., requires_grad=True, dtype=torch.float)

def fun_p0_c(t, h, alpha, beta, gamma, t_min=0):
    x = gamma * ((h**beta) / ((t - t_min)**alpha))
    p0 = 1 / (1 + x)
    div = 1 + x
    return p0, div

################################################################

lr = 0.001
optimizer = optim.Adam([alpha, beta, gamma], lr=lr)


epochs = 6
n_it = 10**1
tau = 1

Loss = []
alphas = []
betas = []
gammas = []

for epoch in range(epochs):
    
    print('Epoch: ', epoch)
    
    gradients = []
    loss = 0

    for name in dict_data_final.keys():

        #print(name)
        x, y = dict_data_final[name]
        grid = Grid(x=x, y=y, mode='gumbel')
        grid.initialize(part=part, inc=inc)
        grid.submatrix()
        grid.p0, grid.div = fun_p0_c(grid.Temp, grid.Hum, alpha=alpha, beta=beta, gamma=gamma)
        grid.enlargement_process_AI()
        grid.montecarlo(n_it=n_it, tau=tau)
        grid.Train = torch.cat((torch.tensor([False]), grid.Train.type(torch.bool)), 0)
        
      
        try:
            probs = get_probs(grid.X0[:, :, grid.Train], grid.X1[:, :, grid.Train], grid.X2[:, :, grid.Train])
            y0 = (grid.y == 0).type(torch.float)
            l = crit_class(probs, grid.y.flatten().long()) + crit_num(grid.X0[:, :, grid.Train], y0)
            loss += l
            l.backward()
            print('Incendio:    ', name, '  Loss: ', l.item())
            print('Gradient: ', alpha.grad.item(), beta.grad.item(), gamma.grad.item())
            gradients.append((alpha.grad.clone(), beta.grad.clone(), gamma.grad.clone()))
        except:
            pass

        alpha.grad.zero_()
        beta.grad.zero_()
        gamma.grad.zero_()

    
    avg_grad_alpha = torch.mean(torch.stack([grad[0] for grad in gradients]), dim=0)
    avg_grad_beta = torch.mean(torch.stack([grad[1] for grad in gradients]), dim=0)
    avg_grad_gamma = torch.mean(torch.stack([grad[2] for grad in gradients]), dim=0)

    print('Los gradientes (alpha, beta, gamma): ', avg_grad_alpha.item(), avg_grad_beta.item(), avg_grad_gamma.item())

    alpha.data -= lr * avg_grad_alpha
    beta.data -= lr * avg_grad_beta
    gamma.data -= lr * avg_grad_gamma

    #print('Los parámetros (alpha, beta, gamma): ', alpha.data.item(), beta.data.item(), gamma.data.item())
    
    print('Los parámetros (alpha, beta, gamma): ', alpha.item(), beta.item(), gamma.item())
    print('Epoch: ', epoch, 'Loss Total: ', loss.item())

    Loss.append(loss.item())
    alphas.append(alpha.item())
    betas.append(beta.item())
    gammas.append(gamma.item())

    gradients = [] # Limpiando la lista de gradientes acumulados

    optimizer.zero_grad() # Limpiando los gradientes acumulados


plt.plot(Loss)
plt.show()

plt.plot(alphas)
plt.plot(betas)
plt.plot(gammas)
plt.show()

print('Los parámetros (alpha, beta, gamma): ', alpha.item(), beta.item(), gamma.item())

# Guardando los parámetros

torch.save(alpha, 'alpha.pt')
torch.save(beta, 'beta.pt')
torch.save(gamma, 'gamma.pt')
