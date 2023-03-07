import pandas as pd
import numpy as np

def Prob_VN(M):
    """ Calculating the probability matrix in the Von Neumman neighbourhood relationship.
    Args:
        M (array): grid in the previous generation.
    Returns:
        array: this array stores the new infected elements which take the value 1.
    """

    N = M.shape[0] 

    id_x = np.where(M==1)[0]; id_y = np.where(M==1)[1]
    indices = list(zip(id_x, id_y))
    
    P = np.full((N+2, N+2), 0)

    for i,j in indices:

        P[i,j+1] = 1
        P[i+2,j+1] = 1
        P[i+1,j] = 1
        P[i+1,j+2] = 1
    
    return P[1:-1, 1:-1]


def Prob_Moore(N, K):
    """Calculating the probability matrix in the Moore neighbourhood relationship.
    Args:
        N (int): size of the grid.
        K (int): number of generation.
    Returns:
        array: this array stores the new infected elements which take the value 1.
    """

    P = np.full((N, N), 0)

    centro = int(N/2)

    inf = -K -1; max = K +2

    for j in range(inf, max):

        P[centro - K -1, centro + j] = 1; P[centro + K + 1, centro + j] = 1
        P[centro + j, centro - K -1] = 1; P[centro + j, centro + K + 1] = 1
    
    return P


def Prob_L(M):
    """Calculating the probability matrix in the chess horse movement relationship.
    Args:
        M (array): grid in the previous generation.
    Returns:
        array: this array stores the new infected elements which take the value 1.
    """

    N = M.shape[0]

    id_x = np.where(M==1)[0]; id_y = np.where(M==1)[1]
    indices = list(zip(id_x, id_y))

    P = np.full((N+4, N+4), 0)

    for i,j in indices:

        P[i, j+1] = 1; P[i, j+3] = 1
        P[i+1, j] = 1; P[i+1, j+4] = 1
        P[i+3, j] = 1; P[i+3, j+4] = 1
        P[i+4, j+1] = 1; P[i+4, j+3] = 1
    
    return P[2:-2, 2:-2]


def VonNeumann(N=7, K=5):
    """ This function carry out the model using Von Neumann neighbourhood.
    Args:
        N (int, optional): size of grid. Defaults to 7.
        K (int, optional): number of generations. Defaults to 5.
    Returns:
        array: shape (N, N, K+1), where the element (:, :, l) is the grid in the l generation.
        dataframe: stores the dynamics of the pest.
    """

    if N%2 == 0:
        N += 1
    
    M = np.full((N,N), 0)
    M[int(N/2), int(N/2)] = 1

    E = np.zeros(N*N*(K+1)).reshape((N,N,K+1))
    E[:,:,0] = M

    columnas = ['Susceptibles', 'Infecteds', 'Deads']
    df = pd.DataFrame(np.array([N**2-1, 1, 0]).reshape(1,3), columns=columnas)
    

    for L in range(K):

        P = Prob_VN(M)
        M[M==2] = 2
        M[M==1] = 2
        M[np.logical_and(M==0, P==1)] = 1
        E[:,:, L+1] = M

        df2 = pd.DataFrame(np.array([np.sum(M==0), np.sum(M==1), np.sum(M==2)]).reshape(1,3), columns=columnas)
        df = pd.concat([df, df2], ignore_index=True)
    
    return E, df


def Moore(N=7, K=5):
    """ This function carry out the model using Moore neighbourhood.
    Args:
        N (int, optional): size of grid. Defaults to 7.
        K (int, optional): number of generations. Defaults to 5.
    Returns:
        array: shape (N, N, K+1), where the element (:, :, l) is the grid in the l generation.
        dataframe: stores the dynamics of the pest.
    """

    if N%2 == 0:
        N += 1
    
    M = np.full((N,N), 0)
    M[int(N/2), int(N/2)] = 1

    E = np.zeros(N*N*(K+1)).reshape((N,N,K+1))
    E[:,:,0] = M

    columnas = ['Susceptibles', 'Infecteds', 'Deads']
    df = pd.DataFrame(np.array([N**2-1, 1, 0]).reshape(1,3), columns=columnas)

    for L in range(K):

        if L <= int(N/2)-1:
            P = Prob_Moore(N,L)
        else:
            P = np.full((N,N), 0)
        

        M[M==2] = 2
        M[M==1] = 2
        M[np.logical_and(M==0, P==1)] = 1
        E[:,:, L+1] = M

        df2 = pd.DataFrame(np.array([np.sum(M==0), np.sum(M==1), np.sum(M==2)]).reshape(1,3), columns=columnas)
        df = pd.concat([df, df2], ignore_index=True)
    
    return E, df


def Caballo(N=7, K=5):
    """ This function carry out the model using chess horse neighbourhood.
    Args:
        N (int, optional): size of grid. Defaults to 7.
        K (int, optional): number of generations. Defaults to 5.
    Returns:
        array: shape (N, N, K+1), where the element (:, :, l) is the grid in the l generation.
        dataframe: stores the dynamics of the pest.
    """

    if N%2 == 0:
        N += 1
    
    M = np.full((N,N), 0)
    M[int(N/2), int(N/2)] = 1

    E = np.zeros(N*N*(K+1)).reshape((N,N,K+1))
    E[:,:,0] = M

    columnas = ['Susceptibles', 'Infecteds', 'Deads']
    df = pd.DataFrame(np.array([N**2-1, 1, 0]).reshape(1,3), columns=columnas)

    for L in range(K):

        P = Prob_L(M)
        M[M==2] = 2
        M[M==1] = 2
        M[np.logical_and(M==0, P==1)] = 1
        E[:,:, L+1] = M

        df2 = pd.DataFrame(np.array([np.sum(M==0), np.sum(M==1), np.sum(M==2)]).reshape(1,3), columns=columnas)
        df = pd.concat([df, df2], ignore_index=True)
    
    return E, df
