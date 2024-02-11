"""
@author: jpzxshi
"""
import numpy as np
import torch

def py2cpp_model(path):
    mionet_py = torch.load(path + '/model_best.pkl')
    
    k = torch.ones(2, 10000, dtype=torch.float64, device=torch.device('cuda'))
    f = torch.ones(2, 10000, dtype=torch.float64, device=torch.device('cuda'))
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64, device=torch.device('cuda'))
    example = (k, f, x)
    
    #mionet_traced = torch.jit.trace(mionet_py, (example,))
    mionet_traced = torch.jit.optimize_for_inference(torch.jit.trace(mionet_py, (example,)))
    print(mionet_traced(example))
    
    mionet_traced.save(path + '/model_best_traced.pt')
    
class Container(torch.nn.Module):
    def __init__(self, values):
        super(Container, self).__init__()
        for key, value in values.items():
            setattr(self, key, value)

def py2cpp_data():
    load_path = './data/Poisson_2d_5000/'
    X_train = np.load(load_path + 'X_train.npz')
    y_train = np.load(load_path + 'y_train.npy')
    X_test = np.load(load_path + 'X_test.npz')
    y_test = np.load(load_path + 'y_test.npy')
    
    save_path = 'C:/xshi/works/codes/workspace_cpp/HIM/HIMProject/data/Poisson_2d_5000/'
    data = {}
    data['X_train'] = (torch.tensor(X_train['arr_0']), torch.tensor(X_train['arr_1']), torch.tensor(X_train['arr_2']))
    data['y_train'] = torch.tensor(y_train)
    data['X_test'] = (torch.tensor(X_test['arr_0']), torch.tensor(X_test['arr_1']), torch.tensor(X_test['arr_2']))
    data['y_test'] = torch.tensor(y_test)
    #torch.save([torch.tensor(X_train['arr_0']), torch.tensor(X_train['arr_1']), torch.tensor(X_train['arr_2'])], to_path + 'X_train.pt')
    #torch.save(torch.tensor(y_train), to_path + 'y_train.pt')
    #torch.save([torch.tensor(X_test['arr_0']), torch.tensor(X_test['arr_1']), torch.tensor(X_test['arr_2'])], to_path + 'X_test.pt')
    #torch.save(torch.tensor(y_test), to_path + 'y_test.pt')
    #print(type(torch.load(to_path + 'X_train.pt')))
    #print(torch.load(to_path + 'X_train.pt')[0].size())
    container = torch.jit.script(Container(data))
    container.save(save_path + 'data.pth')
    

def main():
    #py2cpp_model('./outputs/Poisson_2d_d3_w500_default_lr5_best2/')
    #py2cpp_model('./outputs/Poisson_2d_boundary_d3_w500_default_lr5_50/')
    #py2cpp_data()
    pass

if __name__ == '__main__':
    main()