"""
Some common synthetic benchmark functions for BO
"""
import numpy as np


def reshape(x, input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x

def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx

class hartman_6d():
    '''
    Hartman6 function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 6

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax = 1
        self.name = 'hartman_6d'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2]

        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A = np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
             [2329, 4135, 8307, 3736, 1004, 9991],
             [2348, 1451, 3522, 2883, 3047, 6650],
             [4047, 8828, 8732, 5743, 1091, 381]]

        P = np.asarray(P)
        c = 10**(-4)
        P = np.multiply(P, c)
        outer = 0

        fval = np.zeros((n, 1))
        for idx in range(n):
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = X[idx, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = -(2.58 + outer) / 1.94

        if (n == 1):
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)

class branin():
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(-5, 10), (0, 15)]
        self.fmin = 0.397887
        self.min = [9.424, 2.475]
        self.ismax = 1
        self.name = 'branin'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        a = 1
        b = 5.1/(4*np.pi*np.pi)
        c = 5/np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)
        fx = a*(x2-b*x1*x1+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
        return fx*self.ismax
    
class ackley:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim

        self.bounds =[(0., 1.)]*self.input_dim
    
        self.min = np.zeros(self.input_dim)
        self.fmin = 0
        self.ismax = 1
        self.name = f'ackley{self.input_dim}'
        self._bounds = np.array([(-32.768, 32.768)]*self.input_dim)
        
    def func(self, X_):
        X = reshape(X_, self.input_dim)
        X = from_unit_cube(X, self._bounds[:, 0], self._bounds[:, 1])
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(-1)/self.input_dim))-np.exp(np.cos(2*np.pi*X).sum(-1)/self.input_dim))
        return self.ismax*fval
    
class Levy():
    '''
    Levy function
    '''
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        self.bounds = [(0., 1.)]*self.input_dim
        self.min = [(1.)]*self.input_dim
        self.fmin = 0
        self.ismax = 1
        self.name = f'levy{self.input_dim}'
        self._bounds = np.array([(-10.0, 10.0)]*self.input_dim)

    def func(self, X_):
        X = reshape(X_, self.input_dim)
        X =  from_unit_cube(X, self._bounds[:, 0], self._bounds[:, 1])

        w = np.zeros((X.shape[0], self.input_dim))
        for i in range(1, self.input_dim+1):
            w[:, i-1] = 1 + 1/4*(X[:, i-1]-1)

        fval = (np.sin(np.pi*w[:, 0]))**2 + ((w[:, self.input_dim-1]-1)**2)*(1+(np.sin(2*np.pi*w[:, self.input_dim-1]))**2)
        for i in range(1, self.input_dim):
            fval += ((w[:, i]-1)**2)*(1+10*(np.sin(np.pi*w[:, i]))**2) 

        return self.ismax*fval
    
class rosenbrock():
    '''
    rosenbrock function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        self.bounds = [(0., 1.)]*self.input_dim
        self.min = [(0.)]*self.input_dim
        self.fmin = 0
        self.ismax = 1
        self.name = f'rosenbrock{self.input_dim}'
        self._bounds = np.array([(-2.048, 2.048)]*self.input_dim)
    
    def func(self, X_):
        X = reshape(X_, self.input_dim)
        X =  from_unit_cube(X, self._bounds[:, 0], self._bounds[:, 1])

        fval = 0
        for i in range(self.input_dim-1):
            fval += (100*(X[:, i+1]-X[:, i]**2)**2 + (X[:, i]-1)**2)
        
        return self.ismax*fval

class alpine():
    '''
    alpine function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, sd=None):
        self.input_dim = input_dim
        self.bounds = [(0., 1.)]*self.input_dim
        self.min = [(0)]*input_dim
        self.fmin = 0
        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

        self.ismax = 1
        self.name = f'alpine{self.input_dim}'
        self._bounds = np.array([(-10.0, 10.0)]*input_dim)
        
    def func(self, X_):
        X = reshape(X_, self.input_dim)
        X = from_unit_cube(X, self._bounds[:, 0], self._bounds[:, 1])
        temp = abs(X*np.sin(X) + 0.1*X)
        if len(temp.shape) <= 1:
            fval = np.sum(temp)
        else:
            fval = np.sum(temp, axis=1)

        return self.ismax*fval

class schaffer_n2():
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(-100, 100), (-100, 100)]
        self.fmin = 0
        self.min = [0, 0]
        self.ismax = 1
        self.name = 'schaffer-n2'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        
        fx = 0.5 + (np.sin(x1**2 - x2**2)**2 - 0.5)/(1+0.001*(x1**2+x2**2))**2

        return fx*self.ismax
    
class bohachevsky_n1():
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(-100, 100), (-100, 100)]
        self.fmin = 0
        self.min = [0, 0]
        self.ismax = 1
        self.name = 'bohachevsky-n1'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        
        fx = x1**2 + 2*x2**2 - 0.3* np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7

        return fx*self.ismax

class rastrigin():
    '''
    rastrigin function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        self.bounds = [(0., 1.)]*self.input_dim
        self.min = np.zeros(self.input_dim) 
        self.fmin = 0
        self.ismax = 1
        self.name = f'rastrigin{self.input_dim}'
        self._bounds = np.array([(-5.12, 5.12)]*self.input_dim)

    def func(self, X_):
        X = reshape(X_, self.input_dim) 
        X = from_unit_cube(X, self._bounds[:, 0], self._bounds[:, 1])

        fval = 0
        for i in range(self.input_dim):
            fval += (X[:, i]**2 - 10*np.cos(2*np.pi*X[:, i]))
        fval += 10*self.input_dim
        
        return self.ismax*fval 

class powell():
    '''
    alpine function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None):
        if bounds is None:
            self.bounds = [(-4, 5)]*input_dim
        else:
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim

        self.ismax = 1
        self.name = 'powell'

    def func(self, X):
        assert X.ndim == 1
        X = reshape(X, self.input_dim)
        fval = self.powell(X)
        return self.ismax*fval
    
    def powell(self, x):
        x = np.asarray_chkfinite(x)
        n = self.input_dim
        n4 = ((n + 3) // 4) * 4
        if n < n4:
            x = np.append( x, np.zeros( n4 - n ))
        x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
        f = np.empty_like( x )
        f[0] = x[0] + 10 * x[1]
        f[1] = np.sqrt(5) * (x[2] - x[3])
        f[2] = (x[1] - 2 * x[2]) **2
        f[3] = np.sqrt(10) * (x[0] - x[3]) **2
        return np.sum( f**2 )
    
class ellipsoid():
    '''
    Axis Parallel Hyper-Ellipsoid function 

    '''
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-10.,10.)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = np.zeros(self.input_dim)
        self.fmin = 0
        self.ismax = 1
        self.name = 'ellipsoid'
        
    def func(self,X):
        X = reshape(X,self.input_dim)

        fval = np.array([self.eval_one(x) for x in X])
      
        return self.ismax*fval
    
    def eval_one(self, x):
        assert x.ndim == 1
        assert len(x) == self.input_dim
        fval = np.sum(np.arange(1, len(x)+1) * np.square(x))
        return fval
