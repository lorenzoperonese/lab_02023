"""1. matrici e norme """

import numpy as np

#help(np.linalg) # View source
#help (np.linalg.norm)
#help (np.linalg.cond)

n = 2
A = np.array([[1, 2], [0.499, 1.001]])

print ('Norme di A:')
norm1 = ...
norm2 =  ...
normfro = ...
norminf = ...

print('Norma1 = ', norm1, '\n')
print('Norma2 = ', norm2, '\n')
print('Normafro = ', normfro, '\n')
print('Norma infinito = ', norminf, '\n')

cond1 = ...
cond2 = ...
condfro = ...
condinf = ...

print ('K(A)_1 = ', cond1, '\n')
print ('K(A)_2 = ', cond2, '\n')
print ('K(A)_fro =', condfro, '\n')
print ('K(A)_inf =', condinf, '\n')

x = np.ones((2,1))
b = ...

btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2, 0.5]]).T

# Verificare che xtilde è soluzione di A xtilde = btilde (Axtilde)
my_btilde = ...

print ('A*xtilde = ', btilde)
print(np.linalg.norm(btilde-my_btilde,'fro'))

deltax = ...
deltab = ...

print ('delta x = ', deltax)
print ('delta b = ', deltab)


"""2. fattorizzazione lu"""

import numpy as np

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
x = np.ones((4,1))
b = ...

condA = ...

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
from scipy.linalg import lu_factor as LUdec # pivoting
from scipy.linalg import lu as LUfull # partial pivoting

lu, piv = ...

print('lu',lu,'\n')
print('piv',piv,'\n')


# risoluzione di    Ax = b   <--->  PLUx = b 
my_x = ...

print('my_x = \n', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))

# IMPLEMENTAZIONE ALTERNATIVA - 1

P, L, U = ...
print ('A = ', A)
print ('P = ', P)
print ('L = ', L)
print ('U = ', U)
print ('P*L*U = ', np.matmul(P , np.matmul(L, U))) 

print ('diff = ',   np.linalg.norm(A - np.matmul(P , np.matmul(L, U)), 'fro'  ) ) 


# if P != np.eye(n): 
# Ax = b   <--->  PLUx = b  <--->  LUx = inv(P)b  <--->  Ly=inv(P)b & Ux=y : matrici triangolari
# quindi
invP = np.linalg.inv(P)
y = scipy.linalg.solve_triangular(..., lower=True, unit_diagonal=True)
my_x = scipy.linalg.solve_triangular(..., lower=False)

# if P == np.eye(n): 
# Ax = b   <--->  PLUx = b  <--->  PLy=b & Ux=y
# y = scipy.linalg.solve_triangular(np.matmul(P,L) , b, lower=True, unit_diagonal=True)
# my_x = scipy.linalg.solve_triangular(U, y, lower=False)

print('\nSoluzione calcolata: ', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))


"""2.2 Choleski con matrice di Hilbert"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.hilbert)

# crazione dati e problema test
n = 5
A = ...
x = np.ones((n,1))
b = ...

condA = ...

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = ...
print('L:', L, '\n')

print('L.T*L =', scipy.linalg.norm(A-np.matmul(np.transpose(L),L)))
print('err = ', scipy.linalg.norm(A-np.matmul(np.transpose(L),L), 'fro'))

y = ...
my_x = ...
print('my_x = \n ', my_x)

print('norm =', np.linalg.norm(x-my_x, 'fro'))


K_A = np.zeros((6,1))
Err = np.zeros((6,1))

for n in np.arange(5,11):
    # crazione dati e problema test
    A = ...
    x = ...
    b = ...
    
    # numero di condizione 
    K_A[n-5] = ...
    
    # fattorizzazione 
    L = ...
    y = ...
    my_x = ...
    
    # errore relativo
    Err[n-5] = np.linalg.norm(x-my_x, 'fro')/np.linalg.norm(x, 'fro')
  
xplot = np.arange(5,11)

# grafico del numero di condizione vs dim
plt.semilogy(...)
plt.title('CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(...)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()