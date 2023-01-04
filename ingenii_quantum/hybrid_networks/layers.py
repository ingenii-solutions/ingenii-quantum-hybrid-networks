import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import OpflowQNN, CircuitQNN
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, \
    Gradient
from qiskit.utils import QuantumInstance, algorithm_globals
import torch
from torch.autograd import Function
import torch.nn as nn

from .filters import QuantumFilters2D, QuantumFilters3D

algorithm_globals.random_seed = 42


class QuantumFunction2D(Function):
    description = "Forward pass for the quantum filters"
    class_parameters = {
        "data": "tensor (n_samples, num_features,N,N). Input data",
        "qc_class": "QuantumFilters3D object",
        "use_cuda": "Pytorch device (cpu, cuda) ",
        "tol": "tolerance for the masking matrix. All values from the "
        "original data which are smaller than the tolerance are set to 0.",
    }
    required_parameters = {
        "data": "tensor (n_samples, num_features,N,N). Input data",
        "qc_class": "QuantumFilters2D object"
    }
    optional_parameters = {
        "use_cuda": "Pytorch device (cpu, cuda)",
        "tol": "tolerance for the masking matrix. All values from the "
        "original data which are smaller than the tolerance are set to 0."
    }

    def forward(self, data, qc_class, tol=1e-6, use_cuda=False):
        """
        Forward pass computation
            data: tensor (n_samples, num_features,N,N,N). Input data
            qc_class: QuantumFilters3D object
            use_cuda: Pytorch device (cpu, cuda)
            tol: tolerance for the masking matrix. All values from the original
            data which are smaller than the tolerance are set to 0.
        """
        self.quantum_circuit = qc_class

        result = self.quantum_circuit.get_quantum_filters(data, tol)
        if use_cuda:
            result = result.cuda()
        return result


class QuantumLayer2D(torch.nn.Module):  # QuantumFilters2D):
    description = "Hybrid quantum - classical layer definition "
    class_parameters = {
        "shape": "(n,n), n integer. Box size in which the data is split to "
        "apply the quantum filter. The hilbert space is better exploit if we "
        "set n=2^l",
        "num_filters": "Number of quantum features applied to each of the "
        "features of the data. If the input data has shape (n_samples,C,N,N), "
        "the output data has shape (n_samples, C*num_filters,N,N)",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_name": "Name of family of quantum gates. Implemented: G1, G2, "
        "G3, Ising",
        "num_features": "Number of features of the input data",
        "tol": "tolerance for the masking matrix. All values from the original"
        " data which are smaller than the tolerance are set to 0.",
        "load_unitaries_file_name": "If not None, contains the name of the "
        "file to load the stored unitaries",
        "stride": "int. Stride used to move across the data",
        "saved_gates_filename": "File name for saved gates set",
        "saved_qubits_filename": "File name for saved qubit set",
    }
    required_parameters = {
        "shape": "(n,n), n integer. Box size in which the data is split to "
        "apply the quantum filter. The hilbert space is better exploit if we "
        "set n=2^l"
    }
    optional_parameters = {
        "num_filters": "Number of quantum features applied to each of the "
        "features of the data. If the input data has shape (n_samples,C,N,N), "
        "the output data has shape (n_samples, C*num_filters,N,N)",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_name": "Name of family of quantum gates. Implemented: G1, G2, "
        "G3, Ising",
        "num_features": "Number of features of the input data",
        "tol": "tolerance for the masking matrix. All values from the original"
        " data which are smaller than the tolerance are set to 0",
        "load_unitaries_file_name": "If not None, contains the name of the "
        "file to load the stored unitaries",
        "stride": "int. Stride used to move across the data",
        "saved_gates_filename": "File name for saved gates set",
        "saved_qubits_filename": "File name for saved qubit set",
    }

    def __init__(
            self, shape, num_filters=3,  gates_name='G3',
            num_gates=300, num_features=19, tol=1e-6,
            stride=2, shots=4096, backend='torch',
            load_unitaries_file_name=None,
            unitaries_file_name='unitaries.pickle',
            load_gates=False,
            saved_gates_filename='gates_list.pickle',
            saved_qubits_filename='qubits_list.pickle'):
        """
        Creates the QuantumFilters3D class, generates/loads the unitaries.
        """
        super(QuantumLayer2D, self).__init__()
        self.qc_class = QuantumFilters2D(shape,  stride, shots, backend)
        self.backend = backend
        self.shots = shots

        if self.backend == 'torch':
            if load_unitaries_file_name:  # Load unitaries
                self.qc_class.load_unitaries(load_unitaries_file_name)
            else:  # Initialize unitaries
                self.qc_class.generate_unitaries(
                    gates_name=gates_name,
                    num_gates=num_gates,
                    num_filters=num_filters,
                    num_features=num_features,
                    unitaries_file_name=unitaries_file_name
                )
            self.use_cuda = torch.cuda.is_available()
        else:
            if load_gates:
                self.qc_class.load_gates(
                    gates_name=gates_name,
                    saved_gates_filename=saved_gates_filename,
                    saved_qubits_filename=saved_qubits_filename
                )
            else:
                self.qc_class.generate_qc(
                    gates_name=gates_name,
                    num_gates=num_gates,
                    num_filters=num_filters,
                    num_features=num_features,
                    save=True,
                    saved_gates_filename=saved_gates_filename,
                    saved_qubits_filename=saved_qubits_filename
                )
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
        return QuantumFunction2D.apply(
            data, self.qc_class, self.tol, self.use_cuda)


class QuantumFunction3D(Function):
    description = "Forward pass for the quantum filters"
    class_parameters = {
        "data": "tensor (n_samples, num_features,N,N,N). Input data",
        "qc_class": "QuantumFilters3D object",
        "use_cuda": "Pytorch device (cpu, cuda) ",
        "tol": "tolerance for the masking matrix. All values from the "
        "original data which are smaller than the tolerance are set to 0.",
            }
    required_parameters = {
        "data": "tensor (n_samples, num_features,N,N,N). Input data",
        "qc_class": "QuantumFilters3D object"
    }
    optional_parameters = {
        "use_cuda": "Pytorch device (cpu, cuda)",
        "tol": "tolerance for the masking matrix. All values from the "
        "original data which are smaller than the tolerance are set to 0."
    }

    @staticmethod
    def forward(self, data, qc_class, tol=1e-6, use_cuda=False):
        """
        Forward pass computation
            data: tensor (n_samples, num_features,N,N,N). Input data
            qc_class: QuantumFilters3D object
            use_cuda: Pytorch device (cpu, cuda)
            tol: tolerance for the masking matrix. All values from the
            original data which are smaller than the tolerance are set to 0.
        """
        self.quantum_circuit = qc_class

        result = self.quantum_circuit.get_quantum_filters(data, tol)
        if use_cuda:
            result = result.cuda()
        return result


class QuantumLayer3D(nn.Module):
    description = "Hybrid quantum - classical layer definition "
    class_parameters = {
        "shape": "(n,n,n), n integer. Box size in which the data is split to "
        "apply the quantum filter. The hilbert space is better exploit if we "
        "set n=2^l",
        "num_filters": "Number of quantum features applied to each of the "
        "features of the data. If the input data has shape "
        "(n_samples,C,N,N,N), the output data has shape "
        "(n_samples, C*num_filters,N,N,N)",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_name": "Name of family of quantum gates. Implemented: G1, G2, "
        "G3, Ising",
        "num_features": "Number of features of the input data",
        "tol": "tolerance for the masking matrix. All values from the original"
        " data which are smaller than the tolerance are set to 0.",
        "load_unitaries_file_name": "If not None, contains the name of the"
        " file to load the stored unitaries",
        "stride": "int. Stride used to move across the data",
        "saved_gates_filename": "File name for saved gates set",
        "saved_qubits_filename": "File name for saved qubit set",
    }
    required_parameters = {
        "shape": "(n,n,n), n integer. Box size in which the data is split to "
        "apply the quantum filter. The hilbert space is better exploit if we"
        " set n=2^l"
    }
    optional_parameters = {
        "num_filters": "Number of quantum features applied to each of the "
        "features of the data. If the input data has shape "
        "(n_samples,C,N,N,N), the output data has shape "
        "(n_samples, C*num_filters,N,N,N)",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_name": "Name of family of quantum gates. Implemented: G1, G2, "
        "G3, Ising",
        "num_features": "Number of features of the input data",
        "tol": "tolerance for the masking matrix. All values from the original"
        " data which are smaller than the tolerance are set to 0",
        "load_unitaries_file_name": "If not None, contains the name of the "
        "file to load the stored unitaries",
        "stride": "int. Stride used to move across the data",
        "saved_gates_filename": "File name for saved gates set",
        "saved_qubits_filename": "File name for saved qubit set",
    }

    def __init__(
            self, shape, num_filters=3, gates_name='G3', num_gates=300,
            num_features=19, tol=1e-6, stride=2, shots=4096, backend='torch',
            load_unitaries_file_name=None,
            unitaries_file_name='unitaries.pickle', load_gates=False,
            saved_gates_filename='gates_list.pickle',
            saved_qubits_filename='qubits_list.pickle'):
        """
        Creates the QuantumFilters3D class, generates/loads the unitaries.
        """
        super(QuantumLayer3D, self).__init__()
        self.qc_class = QuantumFilters3D(shape,  stride, shots, backend)
        self.backend = backend
        self.shots = shots

        if self.backend == 'torch':
            if load_unitaries_file_name:  # Load unitaries
                self.qc_class.load_unitaries(load_unitaries_file_name)
            else:  # Initialize unitaries
                self.qc_class.generate_unitaries(
                    gates_name=gates_name,
                    num_gates=num_gates,
                    num_filters=num_filters,
                    num_features=num_features,
                    unitaries_file_name=unitaries_file_name
                )
            self.use_cuda = torch.cuda.is_available()
        else:
            if load_gates:
                self.qc_class.load_gates(
                    gates_name=gates_name,
                    saved_gates_filename=saved_gates_filename,
                    saved_qubits_filename=saved_qubits_filename
                )
            else:
                self.qc_class.generate_qc(
                    gates_name=gates_name,
                    num_gates=num_gates,
                    num_filters=num_filters,
                    num_features=num_features,
                    save=True,
                    saved_gates_filename=saved_gates_filename,
                    saved_qubits_filename=saved_qubits_filename
                )
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
        return QuantumFunction3D.apply(
            data, self.qc_class, self.tol, self.use_cuda)


class QuantumFCLayer:
    description = "Quantum fully-connected layer."
    class_parameters = {
        "input_size": "int. Dimension of the input",
        "n_layers": "int. Number of layers of the ansatz quantum circuit",
        "encoding": "str. Name of the data encoding method. Implemented: "
        "Qubit encoding, amplitude encoding and ZZFeatureMap.",
        "ansatz": "int. Number associated to the ansatz quantum circuit. "
        "Implemented: 1-6 corresponding to circuit_10, circuit_9, circuit_15, "
        "circuit_14, circuit_13, circuit_6",
        "observables": "str or list of str. Name of the observables measured "
        "at the end of the circuit. By default 'Z'*nqbits",
        "backend": "Qiskit backent to run the neural network"
    }
    required_parameters = {
        "input_size": "int. Dimension of the input",
    }
    optional_parameters = {
        "n_layers": "int. Number of layers of the ansatz quantum circuit",
        "encoding": "str. Name of the data encoding method. Implemented: "
        "Qubit encoding, amplitude encoding and ZZFeatureMap.",
        "ansatz": "int. Number associated to the ansatz quantum circuit. "
        "Implemented: 1-6 corresponding to circuit_10, circuit_9, circuit_15, "
        "circuit_14, circuit_13, circuit_6",
        "observables": "str or list of str. Name of the observables measured "
        "at the end of the circuit. By default 'Z'*nqbits. If observables='' "
        "then the probabilities are measured at the end of the circuit and "
        "the output has dimension equal to the number of qubits",
        "backend": "Qiskit backent to run the neural network"
    }
    example_parameters = {
        "input_size": 4,
        "n_layers": 2,
        "encoding": "qubit",
        "ansatz": 1,
        "observables": ['ZIII', 'IZII', 'IIZI', 'IIIZ'],
        "backend": "aer_simulator"
    }

    def __init__(self, input_size, n_layers=2, encoding='qubit', ansatz=1,
                 observables=None, backend="aer_simulator_statevector"):
        self.input_size = input_size
        self.n_layers = n_layers
        self.encoding = encoding

        # Calculate the number of qubits needed for the data encoding
        if self.encoding == 'amplitude':
            self.nqbits = int(np.ceil(np.log2(self.input_size)))
        else:
            self.nqbits = self.input_size

        ansatz_names = [
            "circuit_10", "circuit_9", "circuit_15", "circuit_14",
            "circuit_13", "circuit_6"
        ]
        if type(ansatz) != int or ansatz < 1 or ansatz > 6:
            raise NotImplementedError(
                'Choose a quantum ansatz between 1 and 6')
        self.ansatz = ansatz_names[ansatz-1]

        self.observables = \
            "Z"*self.nqbits if observables is None else observables
        self.backend = backend

    ###########################################################################
    # Quantum encodings
    ###########################################################################

    def _qubit_encoding(self, nqbits):
        """
        Generates the circuit that performs qubit encoding.
            nqbits (int): Number of qubits
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_input = QuantumCircuit(nqbits)
        # 1. Ry gates Initialization
        thetas_ry_input = [
            Parameter(f'thetas_input_{str(i)}')
            for i in range(nqbits)
        ]  # qiskit.circuit.ParameterVector('thetas_ry1', nqbits)
        for i, theta_ry_input in enumerate(thetas_ry_input):
            qc_input.ry(theta_ry_input, i)

        return qc_input, thetas_ry_input

    def _ZZFeatureMap_encoding(self, nqbits, n_layers=2):
        """
        Generates the circuit that performs ZZFeatureMap encoding.
            nqbits (int): Number of qubits
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc = ZZFeatureMap(nqbits, reps=n_layers)
        return qc, qc.parameters

    def _amplitude_encoding(self, nfeatures):
        """
        Generates the circuit that performs amplitude encoding.
            nqbits (int): Number of qubits
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        # 1. Amplitude encoding initialization
        qc_input = RawFeatureVector(nfeatures)
        return qc_input, qc_input.parameters

    ###########################################################################
    # Quantum ansatz
    ###########################################################################

    def _circuit_10(self, nqbits, n_layers):

        """
        Generates the quantum circuit corresponding to the circuit_10 ansatz
        (option 1)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """

        qc_ansatz = QuantumCircuit(nqbits)
        # Choose the number of layers of the PQC
        thetas_ry_weight = []
        qbit_list = [
            (nqbits-1, nqbits-2), (nqbits-1, 0)
        ] + [
            (i, i+1) for i in range(0, nqbits-2)
        ]
        for layer in range(n_layers):

            # 2. Phase shifts
            for (qbit1, qbit2) in qbit_list:
                qc_ansatz.cz(qbit1, qbit2)
            qc_ansatz.barrier()

            # 3. Ry gates
            thetas_new = [
                Parameter(f'thetas_ry{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_ry_weight += thetas_new
            for i, theta_new in enumerate(thetas_new):
                qc_ansatz.ry(theta_new, i)

        return qc_ansatz, thetas_ry_weight

    def _circuit_6(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_6 ansatz
        (option 6)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_rx_list = []
        thetas_rz_list = []
        thetas_crx_list = []
        for layer in range(n_layers):
            thetas_rx = [
                Parameter(f'thetas_rx{str(layer+1)}_{str(i)}')
                for i in range(2*nqbits)
            ]
            thetas_rz = [
                Parameter(f'thetas_rz{str(layer+1)}_{str(i)}')
                for i in range(2*nqbits)
            ]
            thetas_crx = [
                Parameter(f'thetas_crx{str(layer+1)}_{str(i)}')
                for i in range((nqbits-1)*nqbits)
            ]
            thetas_rx_list += thetas_rx
            thetas_rz_list += thetas_rz
            thetas_crx_list += thetas_crx

            # First Rx layer
            for i, theta_rx in enumerate(thetas_rx[:nqbits]):
                qc_ansatz.rx(theta_rx, i)
            # First Ry layer
            for i, theta_rz in enumerate(thetas_rz[:nqbits]):
                qc_ansatz.rz(theta_rz, i)

            # Controlled Rx
            l_idx = 0
            for i in range(nqbits-1, -1, -1):  # i = controlled qubits
                for q in range(nqbits-1, -1, -1):
                    if i == q:
                        # Can't link to itself
                        continue
                    qc_ansatz.crx(thetas_crx[l_idx], i, q)
                    l_idx += 1

            # Second Rx layer
            for i, theta_rx in enumerate(thetas_rx[nqbits:]):
                qc_ansatz.rx(theta_rx, i)
            # Second Ry layer
            for i, theta_rz in enumerate(thetas_rz[:nqbits]):
                qc_ansatz.rz(theta_rz, i)

            qc_ansatz.barrier()
        return qc_ansatz, thetas_rx_list + thetas_rz_list + thetas_crx_list

    def _circuit_9(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_9 ansatz
        (option 2)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_rx_weight = []
        qbit_list = [(i, i-1) for i in range(nqbits-1, 0, -1)]

        for layer in range(n_layers):

            # 1. Hadamard
            for q in range(nqbits):
                qc_ansatz.h(q)

            # 2. Phase shifts
            for (qbit1, qbit2) in qbit_list:
                qc_ansatz.cz(qbit1, qbit2)

            # 3. Ry gates
            thetas_new = [
                Parameter(f'thetas_rx{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_rx_weight += thetas_new
            for i, theta_new in enumerate(thetas_new):
                qc_ansatz.rx(theta_new, i)

            qc_ansatz.barrier()
        return qc_ansatz, thetas_rx_weight

    def _circuit_15(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_15 ansatz
        (option 3)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_ry_list = []
        qbit_list = [(0, nqbits-1)] + \
            [(i, i-1) for i in range(nqbits-1, 0, -1)]
        qbit_list2 = [(nqbits-2, nqbits-1), (nqbits-1, 0)] + \
            [(i, i+1) for i in range(0, nqbits-2)]

        for layer in range(n_layers):

            # 1. Ry gates
            thetas_new = [
                Parameter(f'thetas_ry_1_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_ry_list += thetas_new
            for q, theta_new in enumerate(thetas_new):
                qc_ansatz.ry(theta_new, q)

            # 2. CNOTS
            for (qbit1, qbit2) in qbit_list:
                qc_ansatz.cx(qbit2, qbit1)

            # 3. Ry gates
            thetas_new = [
                Parameter(f'thetas_ry_2_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_ry_list += thetas_new
            for i, theta_new in enumerate(thetas_new):
                qc_ansatz.ry(theta_new, i)

            # 4. CNOTS
            for (qbit1, qbit2) in qbit_list2:
                qc_ansatz.cx(qbit2, qbit1)

            qc_ansatz.barrier()

        return qc_ansatz, thetas_ry_list

    def _circuit_14(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_14 ansatz
        (option 4)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_ry_list = []
        thetas_crx_list = []
        qbit_list = [(0, nqbits-1)] + \
            [(i, i-1) for i in range(nqbits-1, 0, -1)]
        qbit_list2 = [(nqbits-2, nqbits-1), (nqbits-1, 0)] + \
            [(i, i+1) for i in range(0, nqbits-2)]

        for layer in range(n_layers):

            # 1. Ry gates
            thetas_new = [
                Parameter(f'thetas_ry_1_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_ry_list += thetas_new
            for q, theta_new in enumerate(thetas_new):
                qc_ansatz.ry(theta_new, q)

            # 2. Controlled Rx
            thetas_new = [
                Parameter(f'thetas_crx_1_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_crx_list += thetas_new
            for theta_new, (qbit1, qbit2) in zip(thetas_new, qbit_list):
                qc_ansatz.crx(theta_new, qbit2, qbit1)

            # 3. Ry gates
            thetas_new = [
                Parameter(f'thetas_ry_2_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_ry_list += thetas_new
            for i, theta_new in enumerate(thetas_new):
                qc_ansatz.ry(theta_new, i)

            # 4. Controlled Rx
            thetas_new = [
                Parameter(f'thetas_crx_2_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_crx_list += thetas_new
            for theta_new, (qbit1, qbit2) in zip(thetas_new, qbit_list2):
                qc_ansatz.crx(theta_new, qbit2, qbit1)

            qc_ansatz.barrier()
        return qc_ansatz, thetas_ry_list + thetas_crx_list

    def _circuit_13(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_13 ansatz
        (option 5)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_ry_list = []
        thetas_crz_list = []
        qbit_list = [(0, nqbits-1)] + \
            [(i, i-1) for i in range(nqbits-1, 0, -1)]
        qbit_list2 = [(nqbits-2, nqbits-1), (nqbits-1, 0)] + \
            [(i, i+1) for i in range(0, nqbits-2)]

        for layer in range(n_layers):
            # 1. Ry gates
            thetas_new = [
                Parameter(f'thetas_ry_1_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_ry_list += thetas_new
            for q in range(nqbits):
                qc_ansatz.ry(thetas_new[q], q)

            # 2. Controlled Rx
            thetas_new = [
                Parameter(f'thetas_crx_1_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_crz_list += thetas_new
            for theta_new, (qbit1, qbit2) in zip(thetas_new, qbit_list):
                qc_ansatz.crz(theta_new, qbit2, qbit1)

            # 3. Ry gates
            thetas_new = [
                Parameter(f'thetas_ry_2_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_ry_list += thetas_new
            for i, theta_new in enumerate(thetas_new):
                qc_ansatz.ry(theta_new, i)

            # 4. Controlled Rx
            thetas_new = [
                Parameter(f'thetas_crx_2_{str(layer+1)}_{str(i)}')
                for i in range(nqbits)
            ]
            thetas_crz_list += thetas_new
            for theta_new, (qbit1, qbit2) in zip(thetas_new, qbit_list2):
                qc_ansatz.crz(theta_new, qbit2, qbit1)

            qc_ansatz.barrier()
        return qc_ansatz, thetas_ry_list + thetas_crz_list

    def create_qnn(self):
        """
        Creates the quantum neural network composed of the quantum encoding,
        que quantum ansatz and the measurements.
        returns:
            (QuantumNN): quantum neural network
        """
        # Data encoding
        if self.encoding == 'qubit':
            qc_input, params_input = self._qubit_encoding(self.nqbits)
        elif self.encoding == 'amplitude':
            nfeatures = 2**self.nqbits
            qc_input, params_input = self._amplitude_encoding(nfeatures)
        elif self.encoding == 'ZZFeatureMap':
            qc_input, params_input = self._ZZFeatureMap_encoding(self.nqbits)
        else:
            raise NotImplementedError('Data encoding method not implemented')

        # Ansatz
        name_to_func = {
            "circuit_10": self._circuit_10,
            "circuit_6": self._circuit_6,
            "circuit_9": self._circuit_9,
            "circuit_15": self._circuit_15,
            "circuit_14": self._circuit_14,
            "circuit_13": self._circuit_13,
        }
        if self.ansatz not in name_to_func:
            raise NotImplementedError(
                f"Quantum Circuit model '{self.ansatz}' not implemented. "
                "Implemented models: " +
                ", ".join(list(name_to_func.keys()))
            )

        qc_ansatz, params_circuit = \
            name_to_func[self.ansatz](self.nqbits, self.n_layers)

        # Define quantum instances (statevector and sample based)
        if type(self.backend) == str:
            qi = QuantumInstance(Aer.get_backend(self.backend))
        else:
            qi = QuantumInstance(self.backend)
        expval = AerPauliExpectation()
        gradient = Gradient()
        # Define state quantum circuit
        qc_state = StateFn(qc_input.compose(qc_ansatz))

        # Convert to QNN class
        if self.observables != "":  # Case 1: Measuring observables
            # 3. Define the observable to measure
            observable = [
                PauliSumOp.from_list([(obs_name, 1)])
                for obs_name in self.observables
            ]
            # Define operators
            operators = ListOp([
                ~StateFn(obs) @ qc_state
                for obs in observable
            ])
            # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID
            # GRADIENT BACKPROP
            return OpflowQNN(
                operators, params_input, params_circuit, expval, gradient, qi,
                input_gradients=True
            )
        else:  # Case 2: Measuring probabilities
            return CircuitQNN(
                qc_input.compose(qc_ansatz), params_input, params_circuit,
                sparse=False, quantum_instance=qi
            )

    def create_layer(self):
        """
        Creates a Quantum fully connected layer and initializes it
        returns:
            Quantum layer
        """
        qnn = self.create_qnn()
        initial_weights = 0.1 * (
            2 * algorithm_globals.random.random(qnn.num_weights) - 1
        )
        return TorchConnector(qnn, initial_weights)
