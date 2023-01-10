import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class Grid:

    def __init__(self, N):

        if N%2 == 0:
            N += 1

        self.N = N
        self.n = int(N/2)
        self.state = torch.zeros(
            (self.N, self.N),
            dtype=torch.uint8
        )
        self.state[self.n, self.n] = 1
        self.ind = [(self.n, self.n)]

        self.cont = self.state.clone()
        self.neigh_prob = torch.zeros(size=(self.N, self.N), dtype=torch.float64)

        self.susceptible = self.N**2 - 1
        self.infected = 1
        self.dead = 0
    
    def __param__(self, inc=1, part=[0.1, 0.5, 0.9], p0=0.25, div=2):

        self.inc = inc # delay steps between infected and dead cells.
        self.partition = part # partition of the rho axis (3 values). 
        self.p0 = p0  # probability of infection when rho <= part[0].
        self.div = div  # loss factor for further cells.
    
    def load_data(self, df):
        
        """
        Load data from a pandas dataframe with columns:
        'Theta', 'Rho', 'Temperature', 'Humidity'
        """
        self.K = df.shape[0]
        self.Theta = torch.from_numpy(df.Theta.values)
        self.Rho = torch.from_numpy(df.Rho.values)
        self.Temp = torch.from_numpy(df.Temperature.values)
        self.Hum = torch.from_numpy(df.Humidity.values)
        self.Q = (self.Theta/(torch.pi/2) + 1).type(torch.int8)
        self.Xi = torch.where(
            torch.logical_or(self.Q == 1, self.Q == 3),
            self.Theta - ((self.Q-1)/2)*torch.pi,
            (self.Q/2)*torch.pi - self.Theta
        )

        self.m = torch.where(
            self.Rho <= self.partition[1],
            torch.where(
                self.Rho <= self.partition[0],
                0,
                1
            ),
            torch.where(
                self.Rho <= self.partition[2],
                2,
                3
            )
        ).type(torch.int8)

        # create dataframe with all the data

        self.df_wind = pd.DataFrame(
            {
                'Theta': self.Theta,
                'Rho': self.Rho,
                'Q': self.Q,
                'Xi': self.Xi,
                'm': self.m
            }
        )

        self.df_other = pd.DataFrame(
            {
                'Temperature': self.Temp,
                'Humidity': self.Hum
            }
        )

        # create dataframe empty with shape (self.K, 3)
        
        self.df_spread = pd.DataFrame(
            index=range(self.K+1),
            columns=['Susceptible', 'Infected', 'Dead']
        )
        self.df_spread.iloc[0] = [self.susceptible, self.infected, self.dead]

    def submatrix(self):

        self.A = torch.stack(
            (
                self.Xi.sin(),
                torch.where(
                    self.Xi <= torch.pi/4,
                    torch.tan(self.Xi),
                    1/torch.tan(self.Xi)
                ),
                self.Xi.cos()
            ),
            dim=1
        )

    def enlargement_process(self):

        self.large_matrices = torch.zeros(
            (7, 7, self.K),
            dtype=torch.float64
        )
        ind1 = (self.m == 0)
        
        # when rho <= part[0]

        self.large_matrices[2:5, 2:5, ind1] = self.p0
        self.large_matrices[3, 3, ind1] = 0

        # when part[0] < rho <= part[1]

        ind2 = (self.m >= 1)
        self.large_matrices[2, 3, ind2] = self.A[ind2, 0].clone()
        self.large_matrices[2, 4, ind2] = self.A[ind2, 1].clone()
        self.large_matrices[3, 4, ind2] = self.A[ind2, 2].clone()

        # when part[1] < rho <= part[2]

        ind3 = (self.m >= 2)

        self.large_matrices[1, 3, ind3] = (self.A[ind3, 0]/self.div).clone()
        self.large_matrices[1, 5, ind3] = (self.A[ind3, 1]/self.div).clone()
        self.large_matrices[3, 5, ind3] = (self.A[ind3, 2]/self.div).clone()

        self.large_matrices[1, 4, ind3] = (self.large_matrices[1, 3, ind3] + self.large_matrices[1, 5, ind3])/2
        self.large_matrices[2, 5, ind3] = (self.large_matrices[1, 5, ind3] + self.large_matrices[3, 5, ind3])/2

        # when rho > part[2]

        ind4 = (self.m == 3)

        self.large_matrices[0, 3, ind4] = (self.A[ind4, 0]/(self.div**2)).clone()
        self.large_matrices[0, 6, ind4] = (self.A[ind4, 1]/(self.div**2)).clone()
        self.large_matrices[3, 6, ind4] = (self.A[ind4, 2]/(self.div**2)).clone()

        self.large_matrices[0, 4, ind4] = (self.large_matrices[1, 4, ind4]/self.div).clone()
        self.large_matrices[2, 6, ind4] = (self.large_matrices[2, 5, ind4]/self.div).clone()

        self.large_matrices[0, 5, ind4] = (self.large_matrices[0, 4, ind4] + self.large_matrices[0, 6, ind4])/2
        self.large_matrices[1, 6, ind4] = (self.large_matrices[0, 6, ind4] + self.large_matrices[2, 6, ind4])/2

        # rotate the matrices to the right position

        q2 = (self.Q == 2)
        q3 = (self.Q == 3)
        q4 = (self.Q == 4)

        self.large_matrices[:, :, q2] = torch.transpose(
            torch.rot90(
                self.large_matrices[:, :, q2],
                k=1
            ),
            0,
            1
        )

        self.large_matrices[:, :, q4] = torch.transpose(
            torch.rot90(
                self.large_matrices[:, :, q4],
                k=-1
            ),
            0,
            1
        )

        self.large_matrices[:, :, q3] = torch.rot90(
            self.large_matrices[:, :, q3],
            k = 2
        )
    
    def neighbourhood_relation(self, step):

        padding = torch.zeros(self.N + 6, self.N + 6, dtype=torch.float64)
        for i,j in self.ind:
            padding[i:(i+7), j:(j+7)] += self.large_matrices[:, :, step]
        self.neigh_prob = padding[3:-3, 3:-3].clone()

    def update(self, tau=1):

        self.state[self.cont==self.inc] = 2 # infecte to dead once verified the delay

        # update the state of the cells

        # 1. for every single healthy cell, whose probability of being infected is 0 or higher than one
        #    the cell becomes healthy or infected respectively.

        health_cells = (self.state == 0)
        self.state[torch.logical_and(self.neigh_prob <= 0, health_cells)] = 0
        self.state[torch.logical_and(self.neigh_prob >= 1, health_cells)] = 1

        # 2. for every single healthy cell, whose probability of being infected is between 0 and 1
        #    healthy cell + prob between 0, 1 --> update the state of the cell

        to_inf_cells = torch.logical_and(self.neigh_prob < 1, self.neigh_prob > 0)
        ind_to_inf_cells = torch.logical_and(to_inf_cells, health_cells)

        probs = torch.stack(
            (self.neigh_prob[ind_to_inf_cells], 1 - self.neigh_prob[ind_to_inf_cells]),
            dim=1
        ).log()
        self.state[ind_to_inf_cells] = F.gumbel_softmax(logits=probs, tau=tau, hard=True)[:, 0].to(dtype=torch.uint8)

        self.cont[self.state==1] += 1
        aux = torch.where(self.state==1)
        self.ind = list(zip(aux[0], aux[1]))
        self.susceptible = torch.sum(self.state==0).item()
        self.infected = torch.sum(self.state==1).item()
        self.dead = torch.sum(self.state==2).item()
    
    def Spread(self, seed=0, tau=1):

        torch.random.manual_seed(seed)
        np.random.seed(seed)

        self.S = torch.zeros(self.N, self.N, self.K+1, dtype=torch.uint8)
        self.P = torch.zeros(self.N, self.N, self.K, dtype=torch.float64)
        self.S[:, :, 0] = self.state.clone()

        self.submatrix()
        self.enlargement_process()

        for L in range(self.K):
            self.neighbourhood_relation(step=L)
            self.P[:, :, L] = self.neigh_prob.clone()
            self.update(tau=tau)
            self.S[:, :, L+1] = self.state.clone()
            self.df_spread.iloc[L+1] = [self.susceptible, self.infected, self.dead]
    
    def Clear_Init_State(self):

        self.state = torch.zeros(
            (self.N, self.N),
            dtype=torch.uint8
        )
        self.state[self.n, self.n] = 1
        self.cont = self.state.clone()
        self.neigh_prob = torch.zeros(size=(self.N, self.N), dtype=torch.float64)
        self.ind = [(self.n, self.n)]

        self.susceptible = self.N**2 - 1
        self.infected = 1
        self.dead = 0

        self.S = torch.zeros(self.N, self.N, self.K+1, dtype=torch.uint8)
        self.S[:, :, 0] = self.state.clone()
        self.P = torch.zeros(self.N, self.N, self.K, dtype=torch.uint8)

        self.df_spread = pd.DataFrame(
            index=range(self.K+1),
            columns=['Susceptible', 'Infected', 'Dead']
        )
        self.df_spread.iloc[0] = [self.susceptible, self.infected, self.dead]

    def MonteCarlo(self, n_it=10**3, tau=1):

        # we create our tensors to storage the results

        self.X0 = torch.zeros(self.N, self.N, self.K+1, dtype=torch.float64)
        self.X1 = torch.zeros(self.N, self.N, self.K+1, dtype=torch.float64)
        self.X2 = torch.zeros(self.N, self.N, self.K+1, dtype=torch.float64)
        self.P_MC = torch.zeros(self.N, self.N, self.K, dtype=torch.float64)
        #self.df_MC = pd.DataFrame(
        #    index=range(self.K+1),
        #    columns=['Susceptible', 'Infected', 'Dead']
        #)

        # first iteration using seed = 0
        self.Spread(seed=0, tau=tau)

        self.X0 += (self.S==0).to(torch.float64)
        self.X1 += (self.S==1).to(torch.float64)
        self.X2 += (self.S==2).to(torch.float64)
        self.P_MC += self.P
        self.df_MC = self.df_spread.copy()

        # we have to note that self.A and self.large_matrices are already computed.
        # we only need to compute the neighbourhood relation and update the state
        # for every single random seed

        for s in range(1, n_it):

            torch.random.manual_seed(s)
            np.random.seed(s)

            self.Clear_Init_State()

            for L in range(self.K):
                self.neighbourhood_relation(step=L)
                self.P[:, :, L] = self.neigh_prob.clone()
                self.update(tau=tau)
                self.S[:, :, L+1] = self.state.clone()
                self.df_spread.iloc[L+1] += [self.susceptible, self.infected, self.dead]
            
            self.X0 += (self.S==0).to(torch.float64)
            self.X1 += (self.S==1).to(torch.float64)
            self.X2 += (self.S==2).to(torch.float64)
            self.P_MC += self.P
            self.df_MC += self.df_spread
        
        self.X0 /= n_it
        self.X1 /= n_it
        self.X2 /= n_it
        self.P_MC /= n_it
        self.df_MC /= n_it










        

        



