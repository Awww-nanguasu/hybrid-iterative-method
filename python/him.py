"""
@author: jpzxshi
"""
import numpy as np
import torch
import time
    
def hybrid_iterative_method(k, f, x, u, net, nr=None, mode='GS', e=1e-14, m=1):
    n = k.shape[-1]
    h = 1 / (n - 1)
    
    D = (k[:-2] + 2 * k[1:-1] + k[2:]) / (2 * h) * np.eye(n-2)
    L = (k[1:-1] + k[2:]) / (- 2 * h) * np.eye(n-2, k=-1)
    U = (k[:-2] + k[1:-1]) / (- 2 * h) * np.eye(n-2, k=1)
    A = D + L + U
    b = (f[:-2] + 4 * f[1:-1] + f[2:]) * (h / 6)
    
    u = u[1:-1]
    
    if mode == 'Richardson':
        w = h / 4
        B = w * np.eye(n-2)
    elif mode == 'Jacobi':
        w = 1
        B = w * (2 * h) / (k[:-2] + 2 * k[1:-1] + k[2:]) * np.eye(n-2) # D^{-1}
    elif mode == 'GS':
        w = 1
        B = np.linalg.inv(1 / w * D + L)
    
    t0 = time.time() # timing
    
    its = [0] * m
    for i in range(m):
        u_him = np.zeros_like(u)
        C_norm = np.max(np.abs(u_him - u))
        e_0 = np.linalg.norm(u_him - u)
        r = b
        ####
        Ei = np.sqrt(2 * h) * np.sin(np.pi * h * np.arange(1, n-1).reshape(-1, 1) @ np.arange(1, n-1).reshape(1, -1))
        F = [Ei @ (u - u_him)]
        ####
        while (C_norm > e):
            if nr is not None and its[i] % nr == 0:
                #print('MIONet predicting...')
                if its[i] == 0:
                    resi = f 
                else:
                    resi = np.hstack([2 * r[0] - r[1], r, 2 * r[-1] - r[-2]]) / h
                u_him = u_him + net.predict((k, resi, x), returnnp=True)[1:-1]
            else:
                u_him = u_him + B @ r
            r = b - A @ u_him
            C_norm = np.max(np.abs(u_him - u))
            its[i] = its[i] + 1
            F.append(Ei @ (u - u_him))
            #print(C_norm)
        e_it = np.linalg.norm(u_him - u)
        rate = (e_it / e_0) ** (1 / its[i])
        if nr == None:
            print('Richardson convergence rate: ', rate)
            np.save('./spectrum_original', np.array(F))
        else:
            print('Hybrid convergence rate: ', rate)
            np.save('./spectrum_hybrid', np.array(F))
    
    t1 = time.time() # timing
    return {'average time':(t1 - t0) / m, 'average iterations':np.mean(its)}


def solve_Poisson_1d(k, f):
    n = k.shape[-1]
    h = 1 / (n - 1)
    A0 = (k[:-2] + 2 * k[1:-1] + k[2:]) / (2 * h) * np.eye(n-2)
    A1 = (k[:-2] + k[1:-1]) / (- 2 * h) * np.eye(n-2, k=1)
    A2 = (k[1:-1] + k[2:]) / (- 2 * h) * np.eye(n-2, k=-1)
    A = A0 + A1 + A2
    b = (f[:-2] + 4 * f[1:-1] + f[2:]) * (h / 6)
    u = np.linalg.solve(A, b)
    return np.hstack((np.array([0]), u, np.array([0])))

def test_him():
    #### load data
    X_test = np.load('./outputs/Poisson_1d/X_test.npz')
    y_test = np.load('./outputs/Poisson_1d/y_test.npy')
    
    data = {}
    data['k'] = X_test['arr_0']
    data['f'] = X_test['arr_1']
    data['x'] = X_test['arr_2']
    data['u'] = y_test
    
    net = torch.load('./outputs/Poisson_1d/model_best.pkl')
    
    #### test
    n = 6 # 6  19
    #n = np.random.choice(data['k'].shape[0])
    
    #k = data['k'][n]
    k = np.ones_like(data['k'][n])
    f = data['f'][n]
    x = data['x']
    #u = data['u'][n]
    u = solve_Poisson_1d(k, f)
    
    #u_pred = net.predict((k, f, x), returnnp=True).reshape(-1)
    mode = 'Richardson'
    e = 1e-14
    nr = 12
    m = 1
    
    info_original = hybrid_iterative_method(k, f, x, u, net, nr=None, mode=mode, e=e, m=m)
    info_hybrid = hybrid_iterative_method(k, f, x, u, net, nr=nr, mode=mode, e=e, m=m)
    print('original iterative method' + ' (' + mode + '):')
    print(info_original)
    print('hybrid iterative method' + ' (' + mode + ' + MIONet):')
    print(info_hybrid)
    print('speed up X {}'.format(info_original['average time'] / info_hybrid['average time']))

def main():
    test_him()

if __name__ == '__main__':
    main()