"""
@author: jpzxshi
"""
import learner as ln
from gen_data_1d_poisson import Poisson_1d_data 
from postprocessing_1d import postprocessing

def main():
    device = 'gpu' # 'cpu' or 'gpu'
    #### data
    sensors = 50
    train_num = 5000   #### change data size here
    test_num = 5000
    ##### MIONet
    sizes = [
        [sensors, 100, 100, 100],
        [sensors, -100], # -100 means the last layer is without bias
        [1, 100, 100, 100]
        ]
    activation = 'relu'
    initializer = 'default' # 'Glorot normal'
    ##### training
    lr = 1e-5
    iterations = 100000
    batch_size = None
    print_every = 1000
    
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
    data = Poisson_1d_data(sensors, train_num, test_num)
    net = ln.nn.MIONet_Cartesian(sizes, activation, initializer, bias=False)
    ln.Brain.Init(data, net)
    ln.Brain.Run(**training_args)
    ln.Brain.Restore()
    ln.Brain.Output()
    postprocessing(data, ln.Brain.Best_model(), ln.Brain.Loss_history()[0])
    ln.Brain.End()

if __name__ == '__main__':
    main()
