"""
@author: jpzxshi
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

def postprocessing(data, net, loss):
    n = 0
    
    k = data.X_test_np[0][n]
    f = data.X_test_np[1][n]
    x = data.X_test_np[2]
    
    u = data.y_test_np[n]
    u_pred = net.predict((k, f, x), returnnp=True)
    
    x = x.reshape(-1)

    plt.figure(figsize=[6.4 * 2, 4.8 * 2])
    plt.subplot(221)
    plt.plot(x, k, color='black')
    plt.title('k')
    plt.subplot(222)
    plt.plot(x, f, color='black')
    plt.title('f')
    plt.subplot(223)
    plt.plot(x, u, color='b', label='Reference', zorder=0)
    plt.plot(x, u_pred, color='r', label='MIONet', zorder=1)
    plt.title('Prediction')
    plt.legend()
    plt.subplot(224)
    plt.plot(loss[:, 0], loss[:, 1], color='b', label='Train', zorder=0)
    plt.plot(loss[:, 0], loss[:, 2], color='r', label='Test', zorder=1)
    plt.yscale('log')
    plt.legend()
    plt.savefig('Poisson_1d_MIONet_prediction.pdf')
    
def test_figure():
    X_test = np.load('./outputs/Poisson_1d/X_test.npz')
    y_test = np.load('./outputs/Poisson_1d/y_test.npy')
    
    data = {}
    data['k'] = X_test['arr_0']
    data['f'] = X_test['arr_1']
    data['x'] = X_test['arr_2']
    data['u'] = y_test
    
    print(data['k'].shape)
    print(data['f'].shape)
    print(data['x'].shape)
    print(data['u'].shape)
    
    net = torch.load('./outputs/Poisson_1d/model_best.pkl')
    loss = np.loadtxt('./outputs/Poisson_1d/loss.txt')
    print(loss.shape)
    
    n = 19
    #n = np.random.choice(5000)
    print(n)
    
    k = data['k'][n]
    f = data['f'][n]
    x = data['x']
    
    u = data['u'][n]
    u_pred = net.predict((k, f, x), returnnp=True)
    
    x = x.reshape(-1)

    plt.figure(figsize=[6.4 * 2, 4.8 * 2])
    plt.subplot(221)
    plt.plot(x, k, color='black')
    plt.title(r'$k$', fontsize=16)
    plt.subplot(222)
    plt.plot(x, f, color='black')
    plt.title(r'$f$', fontsize=16)
    plt.subplot(223)
    plt.plot(x, u, color='b', label='Reference', zorder=0)
    plt.plot(x, u_pred, color='r', label='MIONet', zorder=1)
    plt.title(r'Prediction', fontsize=16)
    plt.legend(fontsize=12)
    plt.subplot(224)
    plt.plot(loss[:, 0], loss[:,1], color='b', label='Training loss')
    plt.plot(loss[:, 0], loss[:,2], color='r', label='Test loss')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.savefig('Poisson_1d_MIONet_prediction.pdf')
    
def main():
    test_figure()

if __name__ == '__main__':
    main()