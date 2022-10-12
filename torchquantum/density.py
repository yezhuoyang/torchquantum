import torch
import torch.nn as nn
import numpy as np
import torchquantum.functional as tqf
import copy

from torchquantum.macro import C_DTYPE
from typing import Union, List, Iterable


__all__ = ['DensityMatrix']

class DensityMatrix(nn.Module):

    def __init__(self,n_wires: int):
        """Init function for DensityMatrix class(Density Operator)
        Args:
            n_wires (int): how many qubits for the densityMatrix.
        """        
        super().__init__()

        self.n_wires=n_wires
        """
        For example, when n_wires=3
        matrix[001110] denotes the index of |001><110|
        Set Initial value the density matrix of the pure state |00...00>

        """
        _matrix = torch.zeros(2 ** (2*self.n_wires), dtype=C_DTYPE)
        _matrix[0] = 1 + 0j
        _matrix = torch.reshape(_matrix, [2*self.n_wires]*2)



    def trace(self):
        """Return the trace of the DensityMatrix"""
        return torch.trace(self._matrix)

    def positive_semidefinite(self):
        """Check whether the matrix is positive semidefinite by Sylvester's_criterion"""
        return np.all(np.linalg.eigvals(self._matrix) > 0)

    def check_valid(self):
        """Check whether the matrix has trace 1 and is positive semidefinite"""
        return self.trace()==1 and self.positive_semidefinite()


    def spectral(self):
        """Return the spectral of the DensityMatrix"""
        return list(np.linalg.eigvals(self._matrix))


    def tensor(self,other):
        """Return self tensor other(Notice the order)
        Args:
            other (DensityMatrix: Another density matrix
        """
        self._matrix=torch.kron(self._matrix,other._matrix)



    def expand(self,other):
        """Return other tensor self(Notice the order)
        Args:
            other (DensityMatrix: Another density matrix
        """
        self._matrix=torch.kron(other._matrix,self._matrix)



    def clone_matrix(self,existing_matrix: torch.Tensor):
        self._matrix=existing_matrix.clone()


    def set_matrix(self,matrix:Union[torch.tensor,List]):
        matrix = torch.tensor(matrix, dtype=C_DTYPE).to(self.matrix.device)
        self._matrix=matrix



    def set_from_operator(self,gate,wire):
        """Get the density operator of a gate on single qubit.
        
        """
        return 


    def set_from_state(self,probs,states):
        """Get the density matrix of a mixed state.
           Args:
             probs:List of probability of each state
             states:List of state.
           For example:
             probs:[0.5,0.5],states:[|00>,|11>]
           Then the corresponding matrix is: 0.5|00><00|+0.5|11><11|
            0.5, 0, 0, 0
            0  , 0, 0, 0
            0  , 0, 0, 0
            0 ,  0, 0, 0.5 
            self._matrix[00][00]=self._matrix[11][11]=0.5
        """
        return


    def _add(self, other):
        """Return self + other
        Args:
            other (complex): a complex number.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        if not self._matrix.shape==other._matrix.shape:
            raise("Two density matrix must have the same shape.")
        ret = copy.copy(self)
        ret._matrix = self.matrix + other._matrix
        return ret


    def _multiply(self, other):
        """Return other * self.
        Args:
            other (complex): a complex number.
        """
        ret = copy.copy(self)
        ret._matrix = other * self._matrix
        return ret


    def purity(self):
        """Calculate the purity of the DensityMatrix defined as \gamma=tr(\rho^2)
        """
        return torch.trace(torch.matmul(self._matrix, self._matrix))


    def partial_trace(self,dims:List[int]):
        """Calculate the partial trace of given sub-dimension, return a new density_matrix
        Args:
            dims:The list of sub-dimension
            For example, If we have 3 qubit, the matrix shape is (8,8),
            We want to do partial trace to qubit 0,2, dims=[0,2].
            First, the matrix should be reshped to (2,2,2,2,2,2)
            then we call  np.einsum('ijkiqk->jq', reshaped_dm)
        """
        return False


    @property
    def name(self):
        return self.__class__.__name__

    
    def __repr__(self):
        return f"Density Matrix"



    

    
