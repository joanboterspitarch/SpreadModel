import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool
from spread import *
import time

def Param(N_p=11, K_p=10, inc_p=1, n_iter=10**4, n_cor=5):
    global N, K, n, inc, df, n_it, n_cores
    N = N_p
    K = K_p
    n = int(N_p/2)
    inc = inc_p
    np.random.seed(0)
    df = pd.DataFrame({
    'Theta': np.random.uniform(0, 2*np.pi, 10),
    'Rho': np.random.uniform(0, 1, 10),
    'Humidity': np.random.uniform(1, 100, 10),
    'Temperature': np.random.uniform(10, 50, 10)
    })
    n_it, n_cores = n_iter, n_cor

Param()

grid = Grid(N=N)
grid.__param__()
grid.load_data(df=df)
grid.submatrix()
grid.enlargement_process()

large_matrices = grid.large_matrices.clone()


X0_f = torch.zeros(N, N, K+1, dtype=torch.float64)
X1_f = torch.zeros(N, N, K+1, dtype=torch.float64)
X2_f = torch.zeros(N, N, K+1, dtype=torch.float64)
P_MC_f = torch.zeros(N, N, K, dtype=torch.float64)
df_MC_f = pd.DataFrame(
    np.zeros((K+1, 3)),
    columns=['Susceptible', 'Infected', 'Dead']
)


def Task_MonteCarlo(seed):

    torch.random.manual_seed(seed)
    np.random.seed(seed)

    state = torch.zeros(
    (N, N),
    dtype=torch.uint8
    )
    state[n, n] = 1
    cont = state.clone()
    neigh_prob = torch.zeros(size=(N, N), dtype=torch.float64)
    ind = [(n, n)]

    susceptible = N**2 - 1
    infected = 1
    dead = 0

    S = torch.zeros(N, N, K+1, dtype=torch.uint8)
    S[:, :, 0] = state.clone()
    P = torch.zeros(N, N, K, dtype=torch.float64)

    df_spread = pd.DataFrame(
        np.zeros((K+1, 3), dtype='float64'),
        columns=['Susceptible', 'Infected', 'Dead']
    )
    df_spread.iloc[0] = [susceptible, infected, dead]

    for L in range(K):
        padding = torch.zeros(N + 6, N + 6, dtype=torch.float64)
        for i,j in ind:
            padding[i:(i+7), j:(j+7)] += large_matrices[:, :, L].clone()
        neigh_prob = padding[3:-3, 3:-3].clone()
        P[:, :, L] = neigh_prob.clone()
        state[cont==inc] = 2 # infected to dead once verified the delay

        # update the state of the cells

        # 1. for every single healthy cell, whose probability of being infected is 0 or higher than one
        #    the cell becomes healthy or infected respectively.

        health_cells = (state == 0)
        state[torch.logical_and(neigh_prob <= 0, health_cells)] = 0
        state[torch.logical_and(neigh_prob >= 1, health_cells)] = 1

        # 2. for every single healthy cell, whose probability of being infected is between 0 and 1
        #    healthy cell + prob between 0, 1 --> update the state of the cell

        to_inf_cells = torch.logical_and(neigh_prob < 1, neigh_prob > 0)
        ind_to_inf_cells = torch.logical_and(to_inf_cells, health_cells)

        probs = torch.stack(
            (neigh_prob[ind_to_inf_cells], 1 - neigh_prob[ind_to_inf_cells]),
            dim=1
        ).log()
        state[ind_to_inf_cells] = F.gumbel_softmax(logits=probs, hard=True)[:, 0].to(dtype=torch.uint8)

        cont[state==1] += 1
        aux = torch.where(state==1)
        ind = list(zip(aux[0], aux[1]))
        susceptible = torch.sum(state==0).item()
        infected = torch.sum(state==1).item()
        dead = torch.sum(state==2).item()
        S[:, :, L+1] = state.clone()
        df_spread.iloc[L+1] += [susceptible, infected, dead]

    return [(S==0).to(torch.float64), (S==1).to(torch.float64), (S==2).to(torch.float64), P, df_spread]

if __name__ == '__main__':

    start_time = time.time()    
    with ProcessPoolExecutor(n_cores) as executor:
        results = executor.map(Task_MonteCarlo, range(n_it))
        for S0, S1, S2, P, df_spread in results:
            X0_f += S0
            X1_f += S1
            X2_f += S2
            P_MC_f += P
            df_MC_f += df_spread
    X0_f /= n_it
    X1_f /= n_it
    X2_f /= n_it
    P_MC_f /= n_it
    df_MC_f /= n_it
    print('Terminado en:    ', time.time() - start_time, 'segundos')
    print('Puedes acceder a las variables: X0_f, X1_f, X2_f, P_MC_f, df_MC_f')