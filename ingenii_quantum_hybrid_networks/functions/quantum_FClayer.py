import random
import numpy as np
import collections


import qiskit 
from qiskit import QuantumCircuit, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import BasicAer
from qiskit import transpile, assemble
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.runtime import TorchRuntimeClient, TorchRuntimeResult,HookBase
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, StatePreparation
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient
from qiskit_machine_learning.neural_networks import OpflowQNN, TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

import itertools
from qiskit.quantum_info import Pauli
from qiskit.opflow import *
import pickle
import time

algorithm_globals.random_seed = 42

class QuantumFCLayer:
    description = "Quantum fully-connected layer."
    class_parameters = {
        "input_size":"int. Dimension of the input",
        "n_layers":"int. Number of layers of the ansatz quantum circuit",
        "encoding": "str. Name of the data encoding method. Implemented: Qubit encoding, amplitude encoding and ZZFeatureMap.",
        "ansatz":"int. Number associated to the ansatz quantum circuit. Implemented:1-6 corresponding to circuit_10, circuit_9, circuit_15, circuit_14, circuit_13,circuit_6",
        "obs_name":"str or list of str. Name of the observables measured at the end of the circuit. By default 'Z'*nqbits",
        "backend": "Qiskit backent to run the neural network"
                    }
    required_parameters = {
        "input_size":"int. Dimension of the input",}
    optional_parameters = {
        "n_layers":"int. Number of layers of the ansatz quantum circuit",
        "encoding": "str. Name of the data encoding method. Implemented: Qubit encoding, amplitude encoding and ZZFeatureMap.",
        "ansatz":"int. Number associated to the ansatz quantum circuit. Implemented:1-6 corresponding to circuit_10, circuit_9, circuit_15, circuit_14, circuit_13,circuit_6",
        "obs_name":"str or list of str. Name of the observables measured at the end of the circuit. By default 'Z'*nqbits. If obs_name='' then the probabilities are measured at the end of the circuit and the output has dimension equal to the number of qubits",
        "backend": "Qiskit backent to run the neural network"
    }
    example_parameters = {
        "input_size":"4",
        "n_layers":"2",
        "encoding": "qubit",
        "ansatz":"1",
        "obs_name":"['ZIII', 'IZII','IIZI', 'IIIZ']",
        "backend": "aer_simulator"
    }
    def __init__(self, input_size, n_layers=2, encoding='qubit', ansatz=1,
                 obs_name = None, backend = "aer_simulator_statevector"):
        self.input_size = input_size
        self.n_layers = n_layers
        self.encoding = encoding
        # Calcula the number of qubits needed for the data encoding
        if self.encoding=='amplitude':
            self.nqbits = int(np.ceil(np.log2(self.input_size)))
        else:
            self.nqbits = self.input_size 
        ansatz_names = ["circuit_10", "circuit_9", "circuit_15", "circuit_14", "circuit_13","circuit_6"]
        if type(ansatz)!=int or ansatz<1 or ansatz>6:
            raise NotImplementedError('Choose a quantum ansatz between 1 and 6')
        self.ansatz = ansatz_names[ansatz-1]
        if obs_name==None: # Default observables name
            obs_name = "Z"*self.nqbits
        self.obs_name = obs_name
        self.backend = backend
    
    ################################################################################################################################################
    # Quantum encodings
    ################################################################################################################################################
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
        thetas_ry_input = [qiskit.circuit.Parameter('thetas_input_' + str(i)) for i in range(nqbits)]#qiskit.circuit.ParameterVector('thetas_ry1', nqbits)  
        for i in range(nqbits):
            qc_input.ry(thetas_ry_input[i], i)

        return qc_input,thetas_ry_input

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


    ################################################################################################################################################
    # Quantum ansatz
    ################################################################################################################################################

    def _circuit_10(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_10 ansatz (option 1)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)    
        # Choose the number of layers of the PQC
        thetas_ry_weight = []
        qbit_list = [(nqbits-1, nqbits-2), (nqbits-1,0)] + [(i,i+1) for i in range(0, nqbits-2)]
        for layer in range(n_layers):
            # 2. Phase shifts
            for (qbit1, qbit2) in qbit_list:
                qc_ansatz.cz(qbit1, qbit2)
            qc_ansatz.barrier()

            # 3. Ry gates
            thetas_new = [qiskit.circuit.Parameter('thetas_ry' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_ry_weight += thetas_new
            for i in range(nqbits):
                qc_ansatz.ry(thetas_new[i], i)

        return qc_ansatz, thetas_ry_weight

    def _circuit_6(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_6 ansatz (option 6)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        thetas_rx_list = []
        thetas_rz_list = []
        thetas_crx_list = []
        for layer in range(n_layers):    
            thetas_rx = [Parameter('thetas_rx' +str(layer+1)+ '_'+str(i))  for i in range(2*nqbits)]
            thetas_rz = [Parameter('thetas_rz' + str(layer+1)+'_'+str(i))  for i in range(2*nqbits)]
            thetas_crx = [Parameter('thetas_crx' +str(layer+1)+ '_'+str(i))  for i in range((nqbits-1)*nqbits)]
            thetas_rx_list+=thetas_rx
            thetas_rz_list+=thetas_rz
            thetas_crx_list+=thetas_crx

            # First Rx layer
            for i in range(nqbits):
                qc_ansatz.rx(thetas_rx[i],i)
            # First Ry layer
            for i in range(nqbits):
                qc_ansatz.rz(thetas_rz[i],i)
            # Controlled Rx
            l=0
            for i in range(nqbits-1,-1,-1): # i= controlled qubits
                qubits = list(range(nqbits-1,-1,-1))
                qubits.remove(i)
                for q in qubits:
                    qc_ansatz.crx(thetas_crx[l], i,q)
                    l+=1
            # Second Rx layer
            for i in range(nqbits):
                qc_ansatz.rx(thetas_rx[i+nqbits],i)
            # Second Ry layer
            for i in range(nqbits):
                qc_ansatz.rz(thetas_rz[i+nqbits],i)
            qc_ansatz.barrier()       
        return qc_ansatz, thetas_rx_list + thetas_rz_list + thetas_crx_list
    
    def _circuit_9(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_9 ansatz (option 2)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_rx_weight = []
        qbit_list = [(i,i-1) for i in range(nqbits-1,0,-1)]
        for layer in range(n_layers):
            # 1. Hadamard
            for q in range(nqbits):
                qc_ansatz.h(q)
            # 2. Phase shifts
            for (qbit1, qbit2) in qbit_list:
                qc_ansatz.cz(qbit1, qbit2)

            # 3. Ry gates
            thetas_new = [qiskit.circuit.Parameter('thetas_rx' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_rx_weight += thetas_new
            for i in range(nqbits):
                qc_ansatz.rx(thetas_new[i], i)
            qc_ansatz.barrier()
        return qc_ansatz, thetas_rx_weight
    
    def _circuit_15(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_15 ansatz (option 3)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_ry_list = []
        qbit_list = [(0, nqbits-1)] + [(i,i-1) for i in range(nqbits-1,0,-1)]
        qbit_list2 = [(nqbits-2, nqbits-1), (nqbits-1,0)] + [(i,i+1) for i in range(0, nqbits-2)]
        for layer in range(n_layers):
            # 1. Ry gates
            thetas_new = [qiskit.circuit.Parameter('thetas_ry_1_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_ry_list += thetas_new
            for q in range(nqbits):
                qc_ansatz.ry(thetas_new[q], q)
            # 2. CNOTS
            for (qbit1, qbit2) in qbit_list:
                qc_ansatz.cx(qbit2, qbit1)
            # 3. Ry gates
            thetas_new = [qiskit.circuit.Parameter('thetas_ry_2_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_ry_list += thetas_new
            for i in range(nqbits):
                qc_ansatz.ry(thetas_new[i], i)
            # 4. CNOTS
            for (qbit1, qbit2) in qbit_list2:
                qc_ansatz.cx(qbit2, qbit1)
            qc_ansatz.barrier()
            
        return qc_ansatz, thetas_ry_list
    
    def _circuit_14(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_14 ansatz (option 4)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_ry_list = []
        thetas_crx_list = []
        qbit_list = [(0, nqbits-1)] + [(i,i-1) for i in range(nqbits-1,0,-1)]
        qbit_list2 = [(nqbits-2, nqbits-1), (nqbits-1,0)] + [(i,i+1) for i in range(0, nqbits-2)]
        for layer in range(n_layers):
            # 1. Ry gates
            thetas_new = [qiskit.circuit.Parameter('thetas_ry_1_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_ry_list += thetas_new
            for q in range(nqbits):
                qc_ansatz.ry(thetas_new[q], q)
            # 2. Controlled Rx
            thetas_new = [qiskit.circuit.Parameter('thetas_crx_1_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_crx_list+=thetas_new
            i=0
            for (qbit1, qbit2) in qbit_list:
                qc_ansatz.crx(thetas_new[i],qbit2, qbit1)
                i+=1
            # 3. Ry gates
            thetas_new = [qiskit.circuit.Parameter('thetas_ry_2_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_ry_list += thetas_new
            for i in range(nqbits):
                qc_ansatz.ry(thetas_new[i], i)
            # 4. Controlled Rx
            thetas_new = [qiskit.circuit.Parameter('thetas_crx_2_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_crx_list+=thetas_new
            i=0
            for (qbit1, qbit2) in qbit_list2:
                qc_ansatz.crx(thetas_new[i],qbit2, qbit1)
                i+=1
            qc_ansatz.barrier()
        return qc_ansatz, thetas_ry_list + thetas_crx_list

    def _circuit_13(self, nqbits, n_layers):
        """
        Generates the quantum circuit corresponding to the circuit_13 ansatz (option 5)
            nqbits (int): Number of qubits
            n_layers (int): Number of repeated layers of the circuit
        returns:
            (QuantumCircuit): output quantum circuit
            (list of Parameters): Parameters of the circuit
        """
        qc_ansatz = QuantumCircuit(nqbits)
        thetas_ry_list = []
        thetas_crz_list = []
        qbit_list = [(0, nqbits-1)] + [(i,i-1) for i in range(nqbits-1,0,-1)]
        qbit_list2 = [(nqbits-2, nqbits-1), (nqbits-1,0)] + [(i,i+1) for i in range(0, nqbits-2)]
        for layer in range(n_layers):
            # 1. Ry gates
            thetas_new = [qiskit.circuit.Parameter('thetas_ry_1_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_ry_list += thetas_new
            for q in range(nqbits):
                qc_ansatz.ry(thetas_new[q], q)
            # 2. Controlled Rx
            thetas_new = [qiskit.circuit.Parameter('thetas_crx_1_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_crz_list+=thetas_new
            i=0
            for (qbit1, qbit2) in qbit_list:
                qc_ansatz.crz(thetas_new[i],qbit2, qbit1)
                i+=1
            # 3. Ry gates
            thetas_new = [qiskit.circuit.Parameter('thetas_ry_2_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_ry_list += thetas_new
            for i in range(nqbits):
                qc_ansatz.ry(thetas_new[i], i)
            # 4. Controlled Rx
            thetas_new = [qiskit.circuit.Parameter('thetas_crx_2_' + str(layer+1)+'_'+str(i))  for i in range(nqbits)]
            thetas_crz_list+=thetas_new
            i=0
            for (qbit1, qbit2) in qbit_list2:
                qc_ansatz.crz(thetas_new[i],qbit2, qbit1)
                i+=1
            qc_ansatz.barrier()
        return qc_ansatz, thetas_ry_list + thetas_crz_list
    
    
    def create_qnn(self):
        """
        Creates the quantum neural network composed of the quantum encoding, que quantum ansatz and the measurements.
        returns:
            (QuantumNN): quantum neural network
        """
        # Data encoding
        if self.encoding=='qubit':
            qc_input, params_input = self._qubit_encoding(self.nqbits)
        elif self.encoding=='amplitude':
            nfeatures = 2**self.nqbits
            qc_input, params_input= self._amplitude_encoding(nfeatures)
        elif self.encoding=='ZZFeatureMap':
            qc_input, params_input = self._ZZFeatureMap_encoding(self.nqbits)
        else:
            raise NotImplementedError('Data encoding method not implemented')

        # Ansatz
        if self.ansatz=='circuit_10':
            qc_ansatz, params_circuit = self._circuit_10(self.nqbits, self.n_layers)
        elif self.ansatz=='circuit_6':
            qc_ansatz, params_circuit = self._circuit_6(self.nqbits, self.n_layers)
        elif self.ansatz=='circuit_9':
            qc_ansatz, params_circuit = self._circuit_9(self.nqbits, self.n_layers)
        elif self.ansatz=='circuit_15':
            qc_ansatz, params_circuit = self._circuit_15(self.nqbits, self.n_layers)
        elif self.ansatz=='circuit_14':
            qc_ansatz, params_circuit = self._circuit_14(self.nqbits, self.n_layers)
        elif self.ansatz=='circuit_13':
            qc_ansatz, params_circuit = self._circuit_13(self.nqbits, self.n_layers)
        else:
            raise NotImplementedError('Quantum Circuit model not implemented')

        
        # Define quantum instances (statevector and sample based)
        if type(self.backend)==str:
            qi = QuantumInstance(Aer.get_backend(self.backend))
        else:
            qi = QuantumInstance(self.backend)
        expval = AerPauliExpectation()
        gradient = Gradient()
        # Define state quantum circuit
        qc_state = StateFn(qc_input.compose(qc_ansatz))
        
        # Convert to QNN class
        # Case 1: Measuring observables
        if self.obs_name!="":
            # 3. Define the observable to measure
            observable = [PauliSumOp.from_list([(obs_name, 1)]) for obs_name in self.obs_name]
            # Define operators
            operators = ListOp([~StateFn(obs) @ qc_state for obs in observable])
            # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
            qnn =  OpflowQNN(operators, params_input, params_circuit, expval, gradient, qi,input_gradients=True)  
        # Case 2: Measuring probabilities
        else:
            qnn = CircuitQNN(qc_input.compose(qc_ansatz), params_input, params_circuit,sparse=False,quantum_instance=qi)
        return qnn
    
    def create_layer(self):
        """
        Creates a Quantum fully connected layer and initializes it
        returns:
            Quantum layer
        """
        qnn = self.create_qnn()
        initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn.num_weights) - 1)
        qnn_layer = TorchConnector(qnn, initial_weights)
        return qnn_layer