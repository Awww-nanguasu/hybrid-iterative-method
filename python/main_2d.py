"""
@author: jpzxshi
"""
import learner as ln
from gen_data_2d_poisson import Poisson_2d_data
from postprocessing_2d import postprocessing_Poisson_2d

def main():
    device = 'gpu' # 'cpu' or 'gpu'
    #### data
    #sensors = 50
    #train_num = 5000   #### change data size here
    #test_num = 5000
    path = './data/Poisson_2d_5000/'
    ##### MIONet
    sizes = [
        [100 ** 2] + [500] * 3,
        [100 ** 2, -500], # -500 means the last layer is without bias
        [2] + [500] * 3,
        ]
    activation = 'relu'
    initializer = 'default'
    ##### training
    lr = 1e-5
    iterations = 100
    batch_size = 500
    print_every = 10
    
    training_args = {
        'criterion': 'MSE',
        'optimizer': 'Adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': 'best_only',
        'callback': None,
        'dtype': 'double',
        'device': device,
    }
    
    ln.Brain.Start()
    data = Poisson_2d_data(path)
    net = ln.nn.MIONet_Cartesian(sizes, activation, initializer, bias=False)
    ln.Brain.Init(data, net)
    ln.Brain.Run(**training_args)
    ln.Brain.Restore()
    ln.Brain.Output(data=False)
    postprocessing_Poisson_2d(data, ln.Brain.Best_model(), ln.Brain.Loss_history())
    ln.Brain.End()

if __name__ == '__main__':
    main()