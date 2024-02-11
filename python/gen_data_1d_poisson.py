"""
@author: jpzxshi
"""
import numpy as np
from sklearn import gaussian_process as gp
from scipy import interpolate
import matplotlib.pyplot as plt
import learner as ln

class Gaussian_process:
    '''Generate Gaussian process.
    '''
    def __init__(self, interval, mean, std, length_scale, features):
        self.interval = interval # e.g. [0, 1]
        self.mean = mean # e.g. 0
        self.std = std # e.g. 1
        self.length_scale = length_scale # e.g. 0.3
        self.features = features # e.g. 1000

    def generate(self, num):
        x = np.linspace(self.interval[0], self.interval[1], num=self.features)[:, None]
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + 1e-13 * np.eye(self.features))
        return (L @ np.random.randn(self.features, num)).transpose() * self.std + self.mean # [num, features]

def test_gp():
    interval = [0, 1]
    mean = 1
    std = 0.2
    length_scale = 0.1
    features = 1000
    
    gp = Gaussian_process(interval, mean, std, length_scale, features)
    gps = gp.generate(5)
    
    x = np.linspace(*interval, num=features)
    plt.plot(x, gps[0])
    plt.plot(x, gps[1])
    plt.plot(x, gps[2])
    plt.plot(x, gps[3])
    plt.plot(x, gps[4])
    
    
def solve_Poisson_1d(k, f, n):
    h = 1 / (n - 1)
    x = np.linspace(0, 1, num=n)
    kh, fh = k(x), f(x)
    A0 = (kh[:-2] + 2 * kh[1:-1] + kh[2:]) / (2 * h) * np.eye(n-2)
    A1 = (kh[:-2] + kh[1:-1]) / (- 2 * h) * np.eye(n-2, k=1)
    A2 = (kh[1:-1] + kh[2:]) / (- 2 * h) * np.eye(n-2, k=-1)
    A = A0 + A1 + A2
    b = (fh[:-2] + 4 * fh[1:-1] + fh[2:]) * (h / 6)
    u = np.linalg.solve(A, b)
    return np.hstack((np.array([0]), u, np.array([0])))
    
    

class Poisson_1d_data(ln.Data):
    '''Data for 1d Poisson equation.
    '''
    def __init__(self, sensors, train_num, test_num):
        self.sensors = sensors
        self.train_num = train_num
        self.test_num = test_num
        self.GP_k = Gaussian_process([0, 1], 1, 0.2, 0.1, 1000)
        self.GP_f = Gaussian_process([0, 1], 0, 1, 0.1, 1000)
        np.random.seed(0) ##### seed!
        self.__init_data()
        
    def __init_data(self):
        self.X_train, self.y_train = self.__generate(self.train_num)
        self.X_test, self.y_test = self.__generate(self.test_num)
        
    def __generate(self, num):
        gps_k, gps_f = self.GP_k.generate(num), self.GP_f.generate(num)
        def generate(gp_k, gp_f):
            k = interpolate.interp1d(np.linspace(*self.GP_k.interval, num=self.GP_k.features), gp_k, kind='cubic', copy=False, assume_sorted=True)
            f = interpolate.interp1d(np.linspace(*self.GP_f.interval, num=self.GP_f.features), gp_f, kind='cubic', copy=False, assume_sorted=True)
            u = solve_Poisson_1d(k, f, self.sensors)
            k_sensors = k(np.linspace(*self.GP_k.interval, num=self.sensors))
            f_sensors = f(np.linspace(*self.GP_f.interval, num=self.sensors))
            return np.hstack([k_sensors, f_sensors, u])
        res = np.vstack(list(map(generate, gps_k, gps_f)))
        x = np.linspace(*self.GP_k.interval, num=self.sensors)
        return (res[..., :self.sensors], res[..., self.sensors:self.sensors * 2], x[:, None]), res[..., -self.sensors:]

def test_data():
    sensors = 50
    train_num = 100
    test_num = 100
    
    data = Poisson_1d_data(sensors, train_num, test_num)
    print(data.X_train[0].shape, data.X_train[1].shape, data.X_train[2].shape)
    print(data.y_train.shape)
    print(data.X_test[0].shape, data.X_test[1].shape, data.X_test[2].shape)
    print(data.y_test.shape)
    print(data.y_train[0].dtype)
    x = np.linspace(0, 1, num=sensors)
    plt.plot(x, data.y_train[0])
    
#def save_data():
#    sensors = 30
#    train_num = 1000
#    test_num = 1000
    
#    data = Poisson_1d_data(sensors, train_num, test_num)

def main():
    #test_gp()
    test_data()
    #save_data()
    
    

if __name__ == '__main__':
    main()