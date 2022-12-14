from qiskit.opflow import *
import torch
from torch.autograd import Function

from .filters import QuantumFilters2D

class QuantumFunction2D(Function):
    description = "Forward pass for the quantum filters"
    class_parameters = {
        "data": "tensor (n_samples, num_features,N,N). Input data",
        "qc_class": "QuantumFilters3D object",
        "use_cuda": "Pytorch device (cpu, cuda) ",
        "tol": "tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.",
            }
    required_parameters = {
        "data": "tensor (n_samples, num_features,N,N). Input data",
        "qc_class": "QuantumFilters2D object"
    }
    optional_parameters = {
        "use_cuda": "Pytorch device (cpu, cuda)",
        "tol": "tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0."
    }
    
    @staticmethod
    def forward(self, data, qc_class, tol=1e-6, use_cuda=False):
        """ 
        Forward pass computation 
            data: tensor (n_samples, num_features,N,N,N). Input data
            qc_class: QuantumFilters3D object
            use_cuda: Pytorch device (cpu, cuda)
            tol: tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.
        """
        self.quantum_circuit = qc_class

        result = self.quantum_circuit.get_quantum_filters(data, tol)
        if use_cuda:
            result = result.cuda()
        return result
        
        
        
class QuantumLayer2D(torch.nn.Module): #, QuantumFilters2D):
    description = "Hybrid quantum - classical layer definition "
    class_parameters = {
        "shape": "(n,n), n integer. Box size in which the data is split to apply the quantum filter. The hilbert space is better exploit if we set n=2^l",
        "num_filters": "Number of quantum features applied to each of the features of the data. If the input data has shape (n_samples,C,N,N), the output data has shape (n_samples, C*num_filters,N,N)",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_name": "Name of family of quantum gates. Implemented: G1, G2, G3, Ising",
        "num_features": "Number of features of the input data",
        "tol": "tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.",
        "load_U":"If not None, contains the name of the file to load the stored unitaries",
        "save_U": "If load_U is None, save_U contains the name to store the unitaries",
        "stride": "int. Stride used to move across the data"
    }
    required_parameters = {
        "shape": "(n,n), n integer. Box size in which the data is split to apply the quantum filter. The hilbert space is better exploit if we set n=2^l"
    }
    optional_parameters = {
        "num_filters": "Number of quantum features applied to each of the features of the data. If the input data has shape (n_samples,C,N,N), the output data has shape (n_samples, C*num_filters,N,N)",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_name": "Name of family of quantum gates. Implemented: G1, G2, G3, Ising",
        "num_features": "Number of features of the input data",
        "tol": "tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0",
        "load_U":"If not None, contains the name of the file to load the stored unitaries",
        "save_U": "If load_U is None, save_U contains the name to store the unitaries",
        "stride": "int. Stride used to move across the data"
    }
    
    def __init__(self, shape, num_filters=3,  gates_name='G3', num_gates=300, num_features = 19, tol=1e-6,
                 stride=2, shots=4096, backend='torch', load_U=False, name_U='U.pickle',load_gates=False,
                 name_gates='gates_list.pickle', name_qubits='qubits_list.pickle'  ):
        """
        Creates the QuantumFilters3D class, generates/loads the unitaries.
        """
        super(QuantumLayer2D, self).__init__()
        self.qc_class = QuantumFilters2D(shape,  stride, shots, backend)
        self.backend = backend
        self.shots = shots

        if self.backend=='torch':
            if load_U: # Load unitaries
                self.qc_class.load_unitaries(load_U)
            else: # Initialize unitaries
                self.qc_class.generate_unitaries(gates_name,num_gates, num_filters, num_features, unitaries_file_name=name_U)
            self.use_cuda = torch.cuda.is_available()
        else:
            if load_gates:
                self.qc_class.load_gates(
                    gates_name=gates_name,
                    name_gates=name_gates,
                    name_qubits=name_qubits
                )
            else:
                self.qc_class.generate_qc(gates_name,num_gates, num_filters, num_features,True,name_gates,name_qubits)
            self.use_cuda = False
                
        self.gates_name = gates_name
        self.num_gates = num_gates
        self.num_filters = num_filters
        self.tol = tol
        
        
    def forward(self, data):
        """
        Applies the forward pass of the quantum layer
            data (tensor): input data
        """
        return QuantumFunction2D.apply(data, self.qc_class, self.tol, self.use_cuda)   