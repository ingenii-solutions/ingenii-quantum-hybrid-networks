import numpy as np
import pennylane as qml
import torch
from torch.autograd import Function
import torch.nn as nn

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
                 observables=None, backend="default.qubit"):
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
        
        self.observables = [qml.pauli.string_to_pauli_word(o) for o in self.observables]
        self.backend = backend
        self.dev = qml.device(self.backend, wires=self.nqbits)

    ###########################################################################
    # Quantum encodings
    ###########################################################################

    def _qubit_encoding(self, nqbits, feature_vector):
        """
        Generates the circuit that performs qubit encoding.
            feature_vector (array): input data
            nqbits (int): Number of qubits
        """
        qml.AngleEmbedding(features=feature_vector, wires=range(nqbits), rotation='X')


    def _ZZFeatureMap_encoding(self, nqbits, feature_vector, n_layers=2):
        """
        Generates the circuit that performs ZZFeatureMap encoding.
            feature_vector (array): Input data
            nqbits (int): Number of qubits
            n_layers (int): Number of layers
        """
        for _ in range(n_layers):
            for i in range(nqbits):
                qml.Hadamard(wires = i)
                qml.U1(2*feature_vector[i], wires =i)
            # CNOT + hadamard
            for i in range(1,nqbits):
                if i-2>=0:
                    qml.CNOT(wires=[i-2, i-1])
                qml.CNOT(wires=[i-1, i])
                qml.U1(2*(np.pi - feature_vector[i-1])*(np.pi - feature_vector[i]), wires = i)
            qml.CNOT(wires=[nqbits - 2, nqbits-1])

    def _amplitude_encoding(self, feature_vector):
        """
        Generates the circuit that performs amplitude encoding.
            feature_vector (array): Input data
        """
        qml.AmplitudeEmbedding(features=feature_vector, wires=range(self.nqbits),normalize=True)

    def _QAOA_encoding(self, feature_vector, input_weights):
        """
        Generates the circuit that performs QAOA encoding. Notice that this encoding has trainable parameters
            feature_vector (array): Input data
            weights (array): shape (L,1) for 1 qubit, (L,3) for two qubits and (L, 2*nqbits) otherwise
        """
        qml.QAOAEmbedding(features=feature_vector, weights=input_weights, wires=range(len(feature_vector)))

    ###########################################################################
    # Quantum ansatz
    ###########################################################################

    def _circuit_10(self, nqbits, weights):

        """
        Generates the quantum circuit corresponding to the circuit_10 ansatz
        (option 1)
            nqbits (int): Number of qubits
            weights (array): shape (n_layers, nqbits, 3).
        """
        # Choose the number of layers of the PQC
        qml.StronglyEntanglingLayers(weights=weights, wires=range(nqbits))

    def _circuit_6(self, nqbits, weights):
        """
        Generates the quantum circuit corresponding to the circuit_6 ansatz
        (option 6)
            nqbits (int): Number of qubits
            weights (array): shape (nqbits, [ 1 (Rx1) + 1 (Rz1) + 1 x nqbits -1 (cRx) + 1 (Rx2) + 1 (Rz2)] x n_layers)
        """
        size = weights.shape[1]
        n_layers = int(size/(4 + (nqbits - 1)))

        for layer in range(n_layers):
            # First Rx layer
            for i in range(nqbits):
                qml.RX(weights[i,0 + layer-1], wires = i)
            # First RZ layer
            for i in range(nqbits):
                qml.RZ(weights[i,1 + layer-1], wires = i)

            # Controlled Rx
            l_idx = 0
            for i in range(nqbits-1, -1, -1):  # i = controlled qubits
                for q in range(nqbits-1, -1, -1):
                    if i == q:
                        l_idx += 1
                        # Can't link to itself
                        continue
                    ly = l_idx//nqbits
                    lx = l_idx%nqbits
                    qml.CRX(weights[lx, 1 + ly + layer-1], wires = [i, q])
                    l_idx += 1

            # Second Rx layer
            for i in range(nqbits):
                qml.RX(weights[i,4 + nqbits - 3 + layer-1], wires = i)
            # Second Ry layer
            for i in range(nqbits):
                qml.RZ(weights[i,4 + nqbits - 2 + layer-1], wires = i)


    def _circuit_9(self, nqbits, weights):
        """
        Generates the quantum circuit corresponding to the circuit_9 ansatz
        (option 2)
            nqbits (int): Number of qubits
            weights (array): shape (nqbits,  n_layers)
        """
        n_layers = weights.shape[1]
        qbit_list = [(i, i-1) for i in range(nqbits-1, 0, -1)]

        for layer in range(n_layers):

            # 1. Hadamard
            for q in range(nqbits):
                qml.Hadamard(wires = q)

            # 2. Phase shifts
            for (qbit1, qbit2) in qbit_list:
                qml.CZ(wires = [qbit1, qbit2])

            # 3. Ry gates
            for q in range(nqbits):
                qml.RX(weights[q, layer], wires = q)


    def _circuit_15(self, nqbits, weights):
        """
        Generates the quantum circuit corresponding to the circuit_15 ansatz
        (option 3)
            nqbits (int): Number of qubits
            weights (array): shape (n_layers, nqbits)
        """
        qml.BasicEntanglerLayers(weights=weights, wires=range(nqbits))

    def _circuit_14(self, nqbits, weights):
        """
        Generates the quantum circuit corresponding to the circuit_14 ansatz
        (option 4)
            nqbits (int): Number of qubits
            weights (array): shape (nqbits ,  [1 (Ry1) + 1 (CRx1) + 1 (Ry2) + 1 (CRx2)]*n_layers)
        """
        n_layers = int(weights.shape[1]/4)
        qbit_list = [(0, nqbits-1)] + [(i,i-1) for i in range(nqbits-1,0,-1)]
        qbit_list2 = [(nqbits-2, nqbits-1), (nqbits-1,0)] + [(i,i+1) for i in range(0, nqbits-2)]
        for layer in range(n_layers):
            # 1. Ry gates
            for i in range(nqbits):
                qml.RY(weights[i,0 + layer -1], wires = i)

            # 2. Controlled Rx
            for (qbit1, qbit2) in qbit_list:
                qml.CRX(weights[i,1 + layer -1], wires = [qbit2, qbit1])

            # 3. Ry gates
            for i in range(nqbits):
                qml.RY(weights[i,2 + layer -1], wires = i)

            # 4. Controlled Rx
            for (qbit1, qbit2) in qbit_list2:
                qml.CRX(weights[i,1 + layer -1], wires = [qbit2, qbit1])


    def _circuit_13(self, nqbits, weights):
        """
        Generates the quantum circuit corresponding to the circuit_9 ansatz
        (option 5)
            nqbits (int): Number of qubits
            weights (array): shape (nqbits x [1 (Ry1) + 1 (CRx1) + 1 (Ry2) + 1 (CRx2)]*n_layers)
        """
        n_layers = int(weights.shape[1]/4)
        qbit_list = [(0, nqbits-1)] + [(i,i-1) for i in range(nqbits-1,0,-1)]
        qbit_list2 = [(nqbits-2, nqbits-1), (nqbits-1,0)] + [(i,i+1) for i in range(0, nqbits-2)]
        for layer in range(n_layers):
            # 1. Ry gates
            for i in range(nqbits):
                qml.RY(weights[i,0 + layer -1], wires = i)

            # 2. Controlled Rx
            for (qbit1, qbit2) in qbit_list:
                qml.CRZ(weights[i,1 + layer -1], wires = [qbit2, qbit1])

            # 3. Ry gates
            for i in range(nqbits):
                qml.RY(weights[i,2 + layer -1], wires = i)

            # 4. Controlled Rx
            for (qbit1, qbit2) in qbit_list2:
                qml.CRZ(weights[i,1 + layer -1], wires = [qbit2, qbit1])

    def apply_ansatz(self, weights_layer):
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
        # Apply anstatz
        name_to_func[self.ansatz](self.nqbits, weights_layer)

        # Return measurements
        if len(self.observables)>0:  # Case 1: Measuring observables
            return [qml.expval(obs) for obs in self.observables]
        else:  # Case 2: Measuring probabilities
            return qml.probs(wires=list(range(self.nqbits)))


    def qnn_layer(self,inputs, weights_layers):
        """
        Creates the quantum neural network composed of the quantum encoding,
        que quantum ansatz and the measurements.
        returns:
            (QuantumNN): quantum neural network
        """
        # Data encoding
        if self.encoding == 'qubit':
            self._qubit_encoding(self.nqbits, inputs)
        elif self.encoding == 'amplitude':
            self._amplitude_encoding(inputs)
        elif self.encoding == 'ZZFeatureMap':
            self._ZZFeatureMap_encoding(self.nqbits,inputs)
        else:
            raise NotImplementedError('Data encoding method not implemented')
        
        return self.apply_ansatz(weights_layers)

    def qnn_layer_QAOA(self,inputs, weights_layers, weights_input):
        """
        Creates the quantum neural network composed of the quantum encoding,
        que quantum ansatz and the measurements.
        returns:
            (QuantumNN): quantum neural network
        """
        # Data encoding
        self._QAOA_encoding(inputs, weights_input)
        return self.apply_ansatz(weights_layers)

        
    def create_layer(self, type_layer='keras'):
        """
        Creates a Quantum fully connected layer and initializes it
        returns:
            Quantum layer
        """
        # Define weight shapes
        if self.ansatz=='circuit_10':
            self.weights_shape = (self.n_layers,self.nqbits, 3) 
        elif self.ansatz=='circuit_6':
            self.weights_shape = (self.nqbits,(3 + self.nqbits)*self.n_layers)
        elif self.ansatz=='circuit_9':
            self.weights_shape = (self.nqbits, self.n_layers) 
        elif self.ansatz=='circuit_15':
            self.weights_shape =  ( self.n_layers, self.nqbits) 
        elif self.ansatz=='circuit_14':
            self.weights_shape = (self.nqbits,4*self.n_layers) 
        elif self.ansatz=='circuit_13':
            self.weights_shape = (self.nqbits,4*self.n_layers) 

        # Define weight input shapes
        if self.encoding=='QAOA': 
            if self.nqbits==1:
                self.weights_input_shape = (2,1) # 2 layers
            elif self.nqbits==2:
                self.weights_input_shape = (2,3)
            else:
                self.weights_input_shape = (2,2*self.nqbits)
            shapes = {"weights_input":self.weights_input_shape, "weights_layers": self.weights_shape}
            self.qnode = qml.QNode(self.qnn_layer_QAOA, self.dev)
        else:
            shapes = {"weights_layers": self.weights_shape}#"weights_input":(), 
            self.qnode = qml.QNode(self.qnn_layer, self.dev)
        
        if type_layer=='keras':
            if len(self.observables)==0:
                output_dim = self.nqbits
            else:
                output_dim = len(self.observables)
            qlayer = qml.qnn.KerasLayer(self.qnode, shapes, output_dim=output_dim)
        elif type_layer=='torch':
            if self.encoding=='QAOA': 
                init_method = {"weights_layers": torch.nn.init.normal_, "weights_input": torch.nn.init.normal_}
            else:
                init_method = {"weights_layers": torch.nn.init.normal_}
            qlayer = qml.qnn.TorchLayer(self.qnode, shapes, init_method=init_method)

        return qlayer
