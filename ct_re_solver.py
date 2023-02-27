"""
(Very) lightweight solver for linear rational expectations models in continuous time

Suppose our model is given by the linear ODE

[xdot] = [A_xx   A_xy] [x]
[ydot]   [A_yx   A_yy] [y]

where x is a predetermined 'state' vector of length N, y is vector of 'jump' variables,
and the four blocks A_ form a matrix A. For there to be a unique solution for y given
the state x, there must be N stable eigenvalues (i.e. with negative real part) of A.

Let A = QUQ' be the Schur decomposition, where Q is unitary and U is upper triangular.
Assuming that this is sorted so that the stable eigenvalues of U are at the top left,
we can write.

U = [U_ss   U_su]       Q = [Q_xs   Q_xu]
    [0      U_uu]           [Q_ys   Q_yu]

where the N-by-N upper left block U_ss maps the stable subspace to itself, the N-by-N
Q_xs maps the stable subspace to x, etc.

Given any state x, we can obtain the rotated stable state by solving s = Q_xs^(-1)*x.
Then sdot = U_ss*s, and finally xdot = Q_xs*sdot = Q_xs*U_ss*Q_xs^(-1). 
The jump variables y can be obtained from s by y = Q_yx*s = Q_yx*Q_xs^(-1)*x.

Summing up, our model will have law of motion

xdot = B*x
y = F*x

where B = Q_xs*U_ss*Q_xs^(-1) and F = Q_yx*Q_xs^(-1).
"""

import numpy as np
from scipy import linalg

def solver(A, N):
    """For model (xdot, ydot)' = A*(x, y)' with N-dimensional state x, solve for matrices
    B and F that give the unique stable solution xdot = B*x and y = F*x"""
    
    # obtain Schur decomposition A = QUQ', with stable (negative real part)
    # eigenvalues ordered first in U (note: using real Schur decomposition)
    U, Q, n_neg = linalg.schur(A, sort='lhp')
    assert n_neg == N, ('Fails Blanchard-Kahn condition, '
            f'{N} states but {n_neg} negative eigenvalues')

    # obtain B = Q_xs*U_ss*Q_xs^(-1), F = Q_yx*Q_xs^(-1), transposing
    # twice for both to avoid calculating matrix inverse
    B = np.linalg.solve(Q[:N, :N].T, (Q[:N, :N] @ U[:N, :N]).T).T
    F = np.linalg.solve(Q[:N, :N].T, Q[N:, :N].T).T
    
    return B, F
