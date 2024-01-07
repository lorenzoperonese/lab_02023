import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import lu_factor as LUdec 

# Exercise 1

m = 100
n = 10

A = ...

alpha_test = ...
y = ...

print('alpha test', alpha_test)

ATA = A.T@A
ATy = A.T@y

lu, piv = LUdec(...)
alpha_LU = scipy.linalg.lu_solve((lu,piv), ...)

print('alpha LU', alpha_LU)

L = scipy.linalg.cholesky(ATA)
x = scipy.linalg.solve_triangular(np.transpose(L), ..., lower=True)
alpha_chol = scipy.linalg.solve_triangular(L, ..., lower=False)

print('alpha chol', alpha_chol)

U, s, Vh = scipy.linalg.svd(A)

print('Shape of U:', U.shape)
print('Shape of s:', s.shape)
print('Shape of V:', Vh.T.shape)

alpha_svd = np.zeros(s.shape)

for i in range(n):
  ui = U[:, i]
  vi = Vh[i, :]

  alpha_svd = alpha_svd + ...

print('alpha SVD', alpha_svd)


# Exercise 2
case = 2
m = 10
m_plot = 100

# Grado polinomio approssimante
n = 5

if case==0:
    x = np.linspace(-1,1,m)
    y = np.exp(x/2)
elif case==1:
    x = np.linspace(-1,1,m)
    y = 1/(1+25*(x**2))
elif case==2:
    x = np.linspace(0,2*np.pi,m)
    y = np.sin(x)+np.cos(x)


A = ...

for i in range(n+1):
  A[:, i] = ...
  
U, s, Vh = scipy.linalg.svd(A)


alpha_svd = np.zeros(n+1)

for i in range(n+1):
  ui = U[:, i]
  vi = Vh[i, :]

  alpha_svd = alpha_svd + ...

print(alpha_svd)


x_plot = np.linspace(x[0], x[-1], m_plot)
A_plot = np.zeros((m_plot, n+1))

for i in range(n+1):
  A_plot[:, i] = ...

y_interpolation = ...

plt.plot(x, y, 'o')
plt.plot(x_plot, y_interpolation, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Interpolazione polinomiale di grado {n}')
plt.grid()
plt.show()


res = np.linalg.norm(...)
print('Residual: ', res)