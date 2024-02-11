"""
@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt
    
def postprocessing_Poisson_2d_boundary(data, net, loss):
    k = data.X_test[0][0]
    fg = data.X_test[1][0]
    x = data.X_test[2]
    u = data.y_test[0]
    
    u_pred = net.predict((k, fg, x), returnnp=True).reshape(100, 100)
    k = data.tc_to_np(k).reshape(100, 100)
    fg = data.tc_to_np(fg).reshape(100, 100)
    u = data.tc_to_np(u).reshape(100, 100)
    
    plt.figure(figsize=[6.4 * 2, 4.8 * 2])
    plt.subplot(221)
    plt.imshow(np.rot90(k), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('k')
    
    plt.subplot(222)
    plt.imshow(np.rot90(fg), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('fg')
    
    plt.subplot(223)
    plt.imshow(np.rot90(u), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('u')
    
    plt.subplot(224)
    plt.imshow(np.rot90(u_pred), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('MIONet')
    
    plt.savefig('Poisson_2d_boundary_MIONet_prediction.pdf')
    
def main():
    pass

if __name__ == '__main__':
    main()