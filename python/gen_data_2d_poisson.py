"""
@author: jpzxshi
"""
import os
import numpy as np
from sklearn import gaussian_process as gp
from itertools import product
import matplotlib.pyplot as plt
import learner as ln
from pathos.pools import ProcessPool

class Gaussian_process:
    '''Generate Gaussian process.
    '''
    def __init__(self, intervals, mean, std, length_scale, features, e=1e-13):
        self.intervals = intervals # e.g. [0, 1]
        self.mean = mean # e.g. 0
        self.std = std # e.g. 1
        self.length_scale = length_scale # e.g. 0.3
        self.features = features # e.g. 1000
        self.e = e

    @ln.utils.timing
    def generate(self, num):
        if isinstance(self.intervals[0], list):
            itvs = []
            for interval in self.intervals:
                itvs.append(np.linspace(interval[0], interval[1], num=self.features))
            x = np.array(list(product(*itvs)))
            d = len(self.intervals)
        else:
            x = np.linspace(self.intervals[0], self.intervals[1], num=self.features)[:, None]
            d = 1
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + self.e * np.eye(x.shape[0]))
        res = (L @ np.random.randn(x.shape[0], num)).transpose() * self.std + self.mean # [num, features ** d]
        return res.reshape([num] + [self.features] * d)
    
def test_gp_2d():
    intervals = [[0, 1]] * 2
    mean = 1
    std = 0.2
    length_scale = 0.2
    features = 100
    
    gp = Gaussian_process(intervals, mean, std, length_scale, features)
    gps = gp.generate(5)
    
    print(gps.shape)
    
    plt.imshow(np.rot90(gps[0]), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    
def generate_and_save_gps_2d():
    path = './data/'
    gp_k = Gaussian_process([[0, 1]] * 2, 1, 0.2, 0.2, 100)
    gp_f = Gaussian_process([[0, 1]] * 2, 0, 1, 0.2, 100)
    np.random.seed(0)
    for i in range(10):
        gps = {}
        gps['train_k'], gps['train_f'] = gp_k.generate(500), gp_f.generate(500)
        gps['test_k'], gps['test_f'] = gp_k.generate(500), gp_f.generate(500)
        if not os.path.isdir(path): os.makedirs(path)
        np.savez_compressed(path + 'gps_batch_{}'.format(i), **gps)
        
    ##### test
    gps = np.load(path + 'gps_batch_0.npz')
    
    print(gps['train_k'].shape)
    
    plt.imshow(np.rot90(gps['train_k'][0]), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()


########
########


def solve_Poisson_2d(k, f):
    h = 1 / (k.shape[0] - 1)
    n = k.shape[0] - 2
    A0 = ((4 / 3) * k[1:-1, 1:-1]
          + (1 / 3) * (k[:-2, 2:] + k[2:, :-2])
          + (1 / 2) * (k[1:-1, 2:] + k[1:-1, :-2] + k[2:, 1:-1] + k[:-2, 1:-1])).ravel() * np.eye(n ** 2)
    A1 = ((- 1 / 3) * (k[1:-1, 1:-2] + k[1:-1, 2:-1]) 
          + (- 1 / 6) * (k[:-2, 2:-1] + k[2:, 1:-2]))
    A1 = np.hstack([A1, np.zeros([n, 1])]).ravel()[:, None] * np.eye(n ** 2, k=1)
    A2 = ((- 1 / 3) * (k[1:-2, 1:-1] + k[2:-1, 1:-1])
          + (- 1 / 6) * (k[1:-2, 2:] + k[2:-1, :-2]))
    A2 = np.vstack([A2, np.zeros([1, n])]).ravel()[:, None] * np.eye(n ** 2, k=n)
    A = A0 + A1 + A2 + A1.T + A2.T
    b = h ** 2 * ((1 / 2) * f[1:-1, 1:-1]
         + (1 / 12) * (f[2:, 1:-1] + f[1:-1, 2:] + f[:-2, 2:] + f[:-2, 1:-1] + f[1:-1, :-2] + f[2:, :-2])).ravel()
    #import time
    #t = time.time()
    u = np.linalg.solve(A, b).reshape(n, n)
    #print('np.linalg.solve took {} s'.format(time.time() - t))
    return np.hstack([np.zeros([n + 2, 1]), np.vstack([np.zeros([1, n]), u, np.zeros([1, n])]), np.zeros([n + 2, 1])])

@ln.utils.timing
def test_solver():
    k = np.ones((100, 100))
    f = k
    solve_Poisson_2d(k, f)
    
    
@ln.utils.timing
def generate_data_2d(gps_k, gps_f):
    sensors_per_dim = gps_k.shape[-1]
    def generate(k, f):
        u = solve_Poisson_2d(k, f)
        return np.hstack([k.ravel(), f.ravel(), u.ravel()])
    
    #### multi-thread
    p = ProcessPool(nodes=4)
    res = np.vstack(list(p.map(generate, gps_k, gps_f)))
    #### single thread
    #res = np.vstack(list(map(generate, gps_k, gps_f)))
    
    x1 = np.linspace(0, 1, num=sensors_per_dim)
    x2 = np.linspace(0, 1, num=sensors_per_dim)
    x = np.array(list(product(x1, x2)))
    data_k = res[..., :sensors_per_dim ** 2]
    data_f = res[..., sensors_per_dim ** 2:sensors_per_dim ** 2 * 2]
    data_u = res[..., -sensors_per_dim ** 2:]
    return (data_k, data_f, x), data_u

def generate_and_save_data_2d():
    path = './data/Poisson_2d_5000/'
    path_batches = path + 'batches/'
    if not os.path.isdir(path_batches): os.makedirs(path_batches)
    for i in range(9, 10):
        gps = np.load(path + 'gps/gps_batch_{}.npz'.format(i))
        #### test [:5]
        X_train, y_train = generate_data_2d(gps['train_k'], gps['train_f'])
        X_test, y_test = generate_data_2d(gps['test_k'], gps['test_f'])
        ####
        np.savez_compressed(path_batches + 'X_train_batch_{}'.format(i), *X_train)
        np.save(path_batches + 'y_train_batch_{}'.format(i), y_train)
        np.savez_compressed(path_batches + 'X_test_batch_{}'.format(i), *X_test)
        np.save(path_batches + 'y_test_batch_{}'.format(i), y_test)
    #### test figure
    X_train = np.load(path_batches + 'X_train_batch_0.npz')
    y_train = np.load(path_batches + 'y_train_batch_0.npy')
    n = 2
    k, f = X_train['arr_0'][n].reshape(100, 100), X_train['arr_1'][n].reshape(100, 100)
    u = y_train[n].reshape(100, 100)
    
    plt.figure(figsize=[6.4 * 2, 4.8 * 2])
    plt.subplot(221)
    plt.imshow(np.rot90(k), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('k')
    
    plt.subplot(222)
    plt.imshow(np.rot90(f), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('f')
    
    plt.subplot(223)
    plt.imshow(np.rot90(u), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('u')
    
def merge_and_save_data_2d():
    path = './data/Poisson_2d_5000/'
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(10):
        X_train.append(np.load(path + 'batches/X_train_batch_{}.npz'.format(i)))
        y_train.append(np.load(path + 'batches/y_train_batch_{}.npy'.format(i)))
        X_test.append(np.load(path + 'batches/X_test_batch_{}.npz'.format(i)))
        y_test.append(np.load(path + 'batches/y_test_batch_{}.npy'.format(i)))
    def merge(X):
        data_k, data_f = [], []
        for i in range(10):
            data_k.append(X[i]['arr_0'])
            data_f.append(X[i]['arr_1'])
        return (np.vstack(data_k), np.vstack(data_f), X[0]['arr_2'])
    X_train = merge(X_train)
    y_train = np.vstack(y_train)
    X_test = merge(X_test)
    y_test = np.vstack(y_test)
    
    np.savez_compressed(path + 'X_train', *X_train)
    np.save(path + 'y_train', y_train)
    np.savez_compressed(path + 'X_test', *X_test)
    np.save(path + 'y_test', y_test)
    
def test_data_2d():
    path = './data/Poisson_2d_5000/'
    X_train = np.load(path + 'X_train.npz')
    y_train = np.load(path + 'y_train.npy')
    X_test = np.load(path + 'X_test.npz')
    y_test = np.load(path + 'y_test.npy')
    
    print(X_train['arr_0'].shape, X_train['arr_1'].shape, X_train['arr_2'].shape)
    print(y_train.shape)
    print(X_test['arr_0'].shape, X_test['arr_1'].shape, X_test['arr_2'].shape)
    print(y_test.shape)
    
    n = np.random.choice(X_test['arr_0'].shape[0])
    
    k = X_test['arr_0'][n].reshape(100, 100)
    f = X_test['arr_1'][n].reshape(100, 100)
    u_data = y_test[n].reshape(100, 100)
    u_solve = solve_Poisson_2d(k, f)
    
    print('{}-th point from test data'.format(n))
    print('max error: ', np.max(np.abs(u_data - u_solve)))
    
    plt.figure(figsize=[6.4 * 2, 4.8 * 1])
    plt.subplot(121)
    plt.imshow(np.rot90(u_data), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('u (data)')
    plt.subplot(122)
    plt.imshow(np.rot90(u_solve), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('u (solve)')
    
class Poisson_2d_data(ln.data.Data_MIONet_Cartesian):
    '''Data for 2d Poisson equation.
    '''
    def __init__(self, path):
        super(Poisson_2d_data, self).__init__()
        X_train, X_test = np.load(path + '/X_train.npz'), np.load(path + '/X_test.npz')
        self.X_train = (X_train['arr_0'], X_train['arr_1'], X_train['arr_2'])
        self.y_train = np.load(path + '/y_train.npy')
        self.X_test = (X_test['arr_0'], X_test['arr_1'], X_test['arr_2'])
        self.y_test = np.load(path + '/y_test.npy')


import torch
from scipy import interpolate

class Container(torch.nn.Module):
    def __init__(self, values):
        super(Container, self).__init__()
        for key, value in values.items():
            setattr(self, key, value)
    
def generate_and_save(size):
    path = './data/Poisson_2d_size_{}_{}/'.format(size, size)
    if not os.path.isdir(path): os.makedirs(path)
    
    gp_k = Gaussian_process([[0, 1]] * 2, 1, 0.2, 0.2, 100)
    gp_f = Gaussian_process([[0, 1]] * 2, 0, 1, 0.2, 100)
    #np.random.seed(0)

    n = 10
    k = gp_k.generate(n)
    f = gp_f.generate(n)
    x = np.linspace(0, 1, num=100)
    x_size = np.linspace(0, 1, num=size)
    #x_iter = np.tile(x_size, (size, 1)).T.ravel()
    #y_iter = np.tile(x_size, (size, 1)).ravel()
    ks, fs = [], []
    for i in range(n):
        ks.append(interpolate.RectBivariateSpline(x, x, k[i])(x_size, x_size).reshape(size, size))
        fs.append(interpolate.RectBivariateSpline(x, x, f[i])(x_size, x_size).reshape(size, size))
    K = np.stack(ks)
    F = np.stack(fs)
    print(K.shape, F.shape)
    
    ##### test
    
    plt.figure(figsize=[6.4 * 2, 4.8])
    plt.subplot(121)
    plt.imshow(np.rot90(k[0]), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(np.rot90(K[0]), cmap='rainbow')
    plt.xticks([0, (size - 1) / 2, size - 1], [0, 0.5, 1])
    plt.yticks([0, (size - 1) / 2, size - 1], [1, 0.5, 0])
    plt.colorbar()
    
    #### to torch and save
    k_torch = torch.tensor(K)
    f_torch = torch.tensor(F)
    print(k_torch.size(), k_torch.dtype)
    print(f_torch.size(), f_torch.dtype)
    
    data = {}
    data['k'] = k_torch
    data['f'] = f_torch
    container = torch.jit.script(Container(data))
    container.save(path + 'kf_{}_{}_{}_random.pth'.format(n, size, size))

def main():
    #test_gp_2d()
    #generate_and_save_gps_2d()
    #generate_and_save_data_2d()
    #merge_and_save_data_2d()
    #test_data_2d()
    #data = Poisson_2d_data('./data/Poisson_2d_5000/')
    #data.save('./data/test')
    #data.save('./data/test2/')
    #test_solver()
    
    #generate_and_save(1025)
    pass

if __name__ == '__main__':
    main()