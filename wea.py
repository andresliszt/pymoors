from sympy import Expr, Symbol, Add, Mul, Pow, Matrix
from sympy.core.numbers import One


import cvxpy as cp
import numpy as np

x = Symbol('x')
y = Symbol('y')

# wea = x + x + y +
x = cp.Variable(2)
z = np.ones((2,2))@x


import pdb

pdb.set_trace()
