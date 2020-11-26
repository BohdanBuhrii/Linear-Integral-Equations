import numpy as np
import numpy.linalg as LA
from scipy import integrate




a, b = -1, 1

def K(x, y): return x*y + x**2
def f(x): return 1

lambd = 1
h = 0.1


xx = np.arange(a, b + 0.000001, h).round(3).tolist()
n = len(xx)
A = np.zeros((n, n)).tolist()


def basicFunction(j, x, xx=xx, h=h):
  '''j=0...n-1, базис на [xx[j], xx[j+1]]'''
  n = len(xx)-1

  if j >= 1 and xx[j-1] <= x <= xx[j]:
    return (x - xx[j-1])/h
  elif j <= n-1 and xx[j] <= x <= xx[j+1]:
    return (xx[j+1] - x)/h
  else:
    return 0

for i in range(n):
  for j in range(n):

    def Kxi_li(s): return K(xx[i], s) * basicFunction(j, s)

    (integral, err) = integrate.quad(Kxi_li, a, b)

    A[i][j] = basicFunction(j, xx[i]) - integral

print(np.round(A, 3))

ff = [[f(xx[j])] for j in range(n)]



A = np.array(A, dtype='float')
np.round(A, 3)


ff = np.array(ff, dtype='float')
ff.shape

print(LA.det(A))


c = LA.solve(A, ff)
c



funcs = [lambda x: c[i][0]*basicFunction(i, x) for i in range(n)]
def y_approx(x): return sum(f_(x) for f_ in funcs)


def y(x): return 6*x**2 + 1


print(y(0.543), y_approx(0.543))


[y_approx(x) for x in xx]

