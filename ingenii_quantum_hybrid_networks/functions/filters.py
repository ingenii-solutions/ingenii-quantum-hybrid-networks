import collections
import itertools

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit import BasicAer, Aer, assemble, transpile
from qiskit.quantum_info import Pauli
from qiskit.opflow import *

import numpy as np
import pickle
from random import sample

import torch


class QuantumFiltersBase():
    def __init__(self, n_dimensions: int, shape: tuple=(4,4)):
        self.nqbits = int(np.ceil(np.log2(shape[0]**n_dimensions)) + 1) # Calculate the number of qubits

    gate_names = {
        "G1": ["CNOT", "H", "X"],
        "G2": ["CNOT", "H", "S"],
        "G3": ["CNOT", "H", "T"],
    }

    def _select_gates_qubits(self, gates_name, num_gates):
        """
            Selects random gates from the G1, G2 and G3 family, and the qubits to which the gates are applied to.
            gates_name (str): name of the family of quantum gates. Either G1, G2, G3 or Ising
            num_gates (int): depth of the quantum circuits
        """
        # Select random gate
        gates = self.gate_names[gates_name]
        gates_set = [sample(gates, 1)[0] for _ in range(num_gates)]
        
        qubit_idx = list(range(self.nqbits))
        
        def _add_qubit_set(gate):
            if gate=='CNOT':
                # Select qubit 1 and 2 (different qubits)
                qbit1 = sample(qubit_idx, 1)[0]
                qubit_idx2 = qubit_idx.copy()
                qubit_idx2.remove(qbit1)
                qbit2 = sample(qubit_idx2, 1)[0]
                return [qbit1, qbit2]
            else:
                # Select qubit
                qbit = sample(qubit_idx, 1)[0]
                return [qbit]
            
        # Store qubit list of applied gates
        qubits_set = [
            _add_qubit_set(gate)
            for gate in gates_set
        ]
        return gates_set, qubits_set

    def _apply_G_gates(self, qc, gates_set, qubits_set, measure=False):
        """
        Selects a set of random qubits and random gates from the G1, G2 or G3 family, and stores them in a list.
        These gates form the random quantum circuit.
            qc (QuantumCircuit): quantum circuit
            gates_set (list): List of quantum gates that form the quantum reservoir
            qubits_set (list): List of qubits to which the quantum gates are applied to
            measure (bool): mneasure all the qubits (except the ancilla) at the end of the circuit
        """
        # Apply random gates to random qubits
        for gate, qubit in zip(gates_set, qubits_set):

            if gate == 'CNOT': # For 2-qubit gates
                # Select qubit 1 and 2 (different qubits)
                qbit1, qbit2 = qubit
                # Apply gate to qubits
                qc.cx(qbit1, qbit2) 

            else: # For 1-qubit gates
                # Select qubit
                qbit = qubit[0]

                # Apply gate
                if gate == 'X': 
                    qc.x(qbit) 
                elif gate == 'S':
                    qc.s(qbit) 
                elif gate == 'H':
                    qc.h(qbit) 
                elif gate == 'T':
                    qc.t(qbit) 
                else:
                    raise ValueError("Unknown quantum gate:", gate)
        if measure:
            qc.measure(list(range(self.nqbits-1)), list(range(self.nqbits-1)))
    
    def _apply_ising_gates(self, t: float=10):
        """Applies quantum evolution under the transverse field Ising model. The parameters of the Ising model are chosen according to R. Martínez-Peña et al. (2021). 
        Js ~ N(0.75, 0.1), Jij ~ U(-Js/2, Js/2) and hi = h such that h/Js = 0.1. 
            t (float): time evolution
        returns:
            (QuantumCircuit): quantum circuit representing the evolution of the Ising model"""
        # Define parameters Ising model
        Js = np.random.normal(0.75, 0.1)
        h_over_Js = 0.1
        h = Js*h_over_Js
        
        # Get list of all qubits and pairs of qubits
        qubit_idxs = list(range(self.nqbits))
        # qubit_pairs = list(itertools.combinations(qubit_idxs, 2))
        
        # Define the Ising model operators
        pauli_op = 0
        name_gate = 'I' * self.nqbits
        for q1_idx, q2_idx in itertools.combinations(qubit_idxs, 2): # Interaction operators
            name = name_gate[:q1_idx] + 'Z' + name_gate[q1_idx+1:q2_idx] + 'Z' + name_gate[q2_idx+1:]
            coef = np.random.uniform(-Js/2,Js/2)
            pauli_op += coef*PauliOp(Pauli(name))

        for q_idx in qubit_idxs: # Single qubit operators
            name = name_gate[:q_idx] + 'X' + name_gate[(q_idx+1):]
            coef = h
            pauli_op += coef*PauliOp(Pauli(name))
        
        # Time evolution operator exp(iHt)
        evo_time = Parameter('θ')
        evolution_op = (evo_time*pauli_op).exp_i()
        # Trotterization 
        trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=2, reps=1)).convert(evolution_op)
        bound = trotterized_op.bind_parameters({evo_time: t})
        # Convert the trotterized operator to a quantum circuit
        qc_ham = bound.to_circuit()
        return qc_ham
                      

class QuantumFilters2D(QuantumFiltersBase):
    
    description = "Applies a quantum filter to a 2D image.  Quantum data encoding: Flexible Representation of Quantum Images. Quantum transformation: quantum reservoirs with fixed number of gates. Implemented: G1 = {CNOT, H, X}, G2 = {CNOT, H, S}, and G3={CNOT,H,T} and evolution under the transverse field Ising model. The code is designed to run either in a Qiskit backend or with Pytorch."
    class_parameters = {
        "shape": "(n,n), n integer. Box size in which the data is split to apply the quantum filter. The hilbert space is better exploit if we set n^2=2^l.",
        "stride": "int. Stride used to move across the image.",
        "nqbits": "Number of qubits of the quantum circuits. nqbits = [log2(n^2)] + 1. This parameter can not be chosen by the user.",
        "num_filters": "Number of quantum features applied to each of the features of the data. If the input data has shape (n_samples,num_features,N,N), the output data has shape (n_samples, num_features*num_filters,N,N,N)",
        "num_samples": "Number of samples of the input data",
        "device": "Pytorch device (cpu, cuda). Only used if the backend is torch",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_set": "List of quantum gates that form the quantum reservoir",
        "qubits_set":"List of qubits to which the quantum gates are applied to",
        "U_list_filters": "list of all the quantum unitaries that define the quantum reservoirs. The list has shape (num_filters, num_features, nqbits, nqbits). Only used if backend is torch.",
        "shape_windows": "shape of the sliding windows",
        "shots": "Number of shots per experiment (only Qiskit backends)",
        "backend": "Backend where to run the code. Either Qiskit bachend or torch"
            }
    required_parameters = {}
    optional_parameters = {
        "shape": "(n,n), n integer. Box size in which the data is split to apply the quantum filter. The hilbert space is better exploit if we set n^2=2^l.",  
        "stride": "int. Stride used to move across the image.",
        "shots": "Number of shots per experiment (only Qiskit backends)",
        "backend": "Backend where to run the code. Either Qiskit bachend or torch"
    }
    example_parameters = {
        "shape": (2,2),
        "stride": 2,
        "shots": 4096,
        "backend":"aer_simulator"
    }
    
    def __init__(self, shape: tuple=(4,4), stride:float=1, shots=4096, backend='torch'):

        super().__init__(n_dimensions=2, shape=shape)

        self.shape = shape
        self.stride=stride
        self.num_filters = 1
        self.shots=shots
        self.backend = backend
        if self.backend == 'torch':
            # set CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(int("cuda:0".split(':')[1]))
            else:
                self.device = torch.device("cpu")
            
          
    def _FRQI_encoding(self, box):
        '''
        Flexible representation of Quantum Images encoding. Takes an (nxnxn) box and returns the encoded quantum circuit. 
        This function is only used with Qiskit backends.
            box (np.array): (nxnxn) box to be encoded
        '''
        # check if the box has the correct shape
        assert box.shape==self.shape

        # Calculate binary string of basis qubits
        qbit_list = list(itertools.product([0, 1], repeat=self.nqbits-1))
        coefs = {}
        l=0
        for i in range(box.shape[0]):
            for j in range(box.shape[1]):
                theta = box[i,j]
                q_str = ''.join([str(q) for q in qbit_list[l]])
                coefs[q_str + '0'] = np.cos(theta)/np.sqrt(self.shape[0]**2)
                coefs[q_str + '1'] = np.sin(theta)/np.sqrt(self.shape[0]**2)
                l+=1
        # store them in a dictionary and sort it by keys
        coefs = list(collections.OrderedDict(sorted(coefs.items())).values())

        # Define quantum circuit
        cr = ClassicalRegister(self.nqbits-1)
        qr = QuantumRegister(self.nqbits)
        qc = QuantumCircuit(qr,cr)

        # Calculate initial state
        initial_state = np.zeros(2**self.nqbits)
        coefs =  coefs/np.sqrt(np.sum(np.square(coefs)))
        initial_state[:coefs.shape[0]] = coefs
        qc.initialize(initial_state, list(range(self.nqbits)))
        return qc, initial_state    
    
    
    def _rollNumpy(self, a, b, dx=1, dy=1):   
        '''
        Rolling 3D window for numpy array. This function is only used with Qiskit backends.
            a (np.array): input array
            b (np.array): rolling 2D window array
            dx (int): horizontal step, abscissa, number of columns
            dy (int): vertical step, ordinate, number of rows
        '''
        shape = a.shape[:-2] + \
                ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
                ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
                b.shape  # multidimensional "sausage" with 3D cross-section
        strides = a.strides[:-2] + \
                  (a.strides[-2] * dy,) + \
                  (a.strides[-1] * dx,) + \
                  a.strides[-2:]
        self.shape_windows = shape
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def _roll(self, a, b, dx=1, dy=1):  
        '''
        Rolling 3D window for pytorch tensor. This function is only used with Pytorch backends.
            a (tensor): input array, shape (n_samples, N,N,N)
            b (tensor): rolling 3D window array, shape (n,n,n)
            dx (int): horizontal step, abscissa, number of columns
            dy (int): vertical step, ordinate, number of rows
        '''
        shape = a.shape[:-2] + \
                ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
                ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
                b.shape  # multidimensional "sausage" with 3D cross-section
        strides = a.stride()[:-2] + \
                (a.stride()[-2] * dy,) + \
                (a.stride()[-1] * dx,) + \
                  a.stride()[-2:]

        self.shape_windows = shape
        return torch.as_strided(a, shape, strides)
    
    def _scale_data(self, data):
        """
        Scale the data to [0, pi/2) (each feature is scaled separately)
            data (tensor): input data, shape (n_samples, num_features, N,N,N)
        returns:
            (tensor): scaled data, shape (n_samples, num_features, N,N,N)
        """
        new_min, new_max = 0, np.pi/2
        for i in range(data.shape[0]): # Scale each feature independently
            for j in range(data.shape[1]):
                v_min, v_max = data[i,j,:,:].min(), data[i,j,:,:].max()
                if v_min < v_max:
                    data[i,j,:,:] = (data[i,j,:,:] - v_min)/(v_max - v_min)*(new_max - new_min) + new_min                    
        return data
    
    def _run_boxQiskit(self, box, gates_set, qubits_set):
        '''
        For a given box of the image runs the FRQI encoding + random quantum circuit.  This function is only used with Qiskit backends.
            box (np.array): shape (n,n). Input data
            gates_set (list): Set of random quantum gates
            qubits_set (list): List of qubits to apply the quantum gates
        '''
        # FRQI encoding
        qc, initial_state = self._FRQI_encoding(box)
        
        # Random quantum circuit
        if self.gates_name=='Ising':
            qc_Ising = self._apply_ising_gates()
            qc += qc_Ising
        else:
            self._apply_G_gates(qc, gates_set, qubits_set, measure=True)
        
        # Get counts
        if self.backend=='aer_simulator':
            aer_sim = Aer.get_backend(self.backend)
            t_qc = transpile(qc, aer_sim)
            qobj = assemble(t_qc, shots=self.shots)
            result = aer_sim.run(qobj).result()
        else: # For real hardware/fake simulators
            coupling_map = self.backend.configuration().coupling_map
            optimized_3 = transpile(qc, backend=self.backend, seed_transpiler=11)
            result = self.backend.run(optimized_3, shots=self.shots).result()
        counts = result.get_counts(qc)
        
        # Calculate binary string of basis qubits
        qbit_list = list(itertools.product([0, 1], repeat=self.nqbits-1))
        new_counts = []
        for l in range(len(qbit_list)):
            q_str =''.join([str(value) for value in qbit_list[l]])
            new_counts.append(counts.get(q_str, 0))

        new_counts/=np.sum(new_counts)
        return new_counts
    
    def _run_filterQiskit(self, data, gates_set, qubits_set, tol=1e-6):
        '''
        Runs the quantum circuit for one feature of the data.  This function is only used with Qiskit backends.
            data (np.array): shape (N,N), channel of the image 
            gates_set (list): Set of random quantum gates
            qubits_set (list): List of qubits to apply the quantum gates
            scale (bool): Scale the input image
        '''
        # Get boxes from data
        size = self.shape[0]
        b=np.zeros(self.shape)
        windows = self._rollNumpy(data,b,dx=size*self.stride,dy=size*self.stride).reshape(-1,size,size)       
        
        results = []
        for i in range(windows.shape[0]): # Loop through every box of the image
            # If all values are zero, no need to run the algo
            box = windows[i]
            if np.sum(np.abs(box))>tol: # Run the QC if the box is non-zero valued
                result = np.array(self._run_boxQiskit(box, gates_set, qubits_set))
                result = result[:self.shape[0]**2].reshape(self.shape)
            else: # Return zeros if the box is all zero-valued
                result = np.array([0 for i in range(self.shape[0]**2)]).reshape(self.shape)
            results.append(result)
        
        # Reshape the results in the original shape
        results = np.array(results).reshape(self.shape_windows) # Reshape results to rolling windows shape
        fin_shape = int(data.shape[-1]/self.stride)
        results = results.transpose([0, 2, 1, 3]).reshape(fin_shape,fin_shape)# Transpose to obtain original shape
        
        # Apply zero-mask
        mask = np.ones(self.shape_windows) 
        windows_reshape = windows.reshape(self.shape_windows)# Reshape to rolling windows shape
        mask[np.abs(windows_reshape)<tol] = 0 #Find zero-valued values of the original data
        mask = mask.reshape(self.shape_windows).transpose([0, 2, 1, 3]).reshape((fin_shape,fin_shape))# Transpose to return to original shape
        results = results.astype('float64')
        results*=mask #Apply mask
        return results
    
    def _run_filter(self, data,tol, U):
        """
            Runs a quantum filter to a feature from the data samples.  This function is only used with Pytorch backend.
                data (tensor): input data (one feature), shape (num_samples, N,N,N)
                tol (float): tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.
                U (tensor): Unitary matrix representing the random quantum circuit
            returns:
                (tensor): quantum filter applied to data
        """
        # 0. Get rolling windows
        size = self.shape[0]
        b=torch.zeros(self.shape)
        windows = self._roll(data,b,dx=size*self.stride,dy=size*self.stride).reshape(-1,size,size).to(self.device)


        # 1. Flexible representation of quantum images. Apply cos(theta)|0> + sin(theta)|1> transformation
        v1 = torch.tensor([1,0]).to(self.device)
        v2 = torch.tensor([0,1]).to(self.device)
        sq = torch.tensor([self.shape[0]**2]).to(self.device)

        frqi = (torch.kron(torch.cos(windows),  v1) +
           torch.kron(torch.sin(windows) , v2))/torch.sqrt(sq)

        # 2. Flatten the last dimension to create an initial state
        flatten = frqi.reshape(-1,2**(self.nqbits)).type(torch.complex128)
        # 3. Apply U rotation to each state
        apply_U = (U @ flatten.T).reshape(-1,size, size, 2)

        # 4. Partial trace to remove the extra qubit
        partial_trace = torch.einsum('ijkl->ijk', apply_U)

        # 5. Reshape back to window view
        partial_trace_reshaped = partial_trace.reshape(self.shape_windows)

        # 6. Reshape to original shape, taking into account transposition
        fin_shape = int(data.shape[-1]/self.stride)
        output = partial_trace_reshaped.permute([0, 1, 3, 2, 4]).reshape((self.num_samples, 
                                                                 fin_shape,fin_shape))

        # 7. Get the probability for each state (modulus of the final state)
        output_probability = (output.conj()*output).real

        # 8. Create a mask to map to zero all zero values of the original image
        mask = torch.ones(windows.shape)
        mask[np.abs(windows)<tol] = 0
        mask = mask.reshape(self.shape_windows).permute([0, 1, 3, 2, 4]).reshape((self.num_samples,
                                                                                      fin_shape,fin_shape)).to(self.device)
        quantum_filter = output_probability*mask        
        return quantum_filter
        
    
    def generate_unitaries(self, gates_name='G3', num_gates=300, num_filters=3, num_features=19, unitaries_file_name='unitaries.pickle', save=True):
        """
        Generates the quantum unitaries that represent the quantum reservoirs (random quantum circuits).  This function is only used with Pytorch backend.
            gates_name (str): name of the family of quantum gates. Either G1, G2, G3 or Ising
            num_gates (int): depth of the quantum circuits
            num_filters (int): Number of quantum filters to apply to each feature
            num_features (int): Number of features of the data
            unitaries_file_name (str): name of the file containing unitary list (only needed if save=True)
            save (bool): if True, the generated unitaries are saved to a pickle file.
        """
        if self.backend != 'torch':
            raise ValueError("This function is only callable with the 'torch' backend")
        self.U_list_filters = []
        self.num_filters = num_filters
        self.gates_name = gates_name
        for i in range(num_filters):
            U_list = []
            for j in range(num_features):
                # Store circuit parameters
                self.gates_name = gates_name
                self.num_gates = num_gates
                # If the quantum reservoir is the Ising model
                if gates_name=='Ising':
                    qc = self._apply_ising_gates()
                else: # Otherwise, if it is one of the G families
                    gates_set, qubits_set = self._select_gates_qubits(gates_name, num_gates)
                    # Get unitary
                    qc = QuantumCircuit(self.nqbits)
                    # Random quantum circuit
                    self._apply_G_gates(qc, gates_set, qubits_set)
                # Get unitary
                backend = BasicAer.get_backend('unitary_simulator')
                job = backend.run(transpile(qc, backend))
                U = job.result().get_unitary(qc)
                U_list.append(U)
            self.U_list_filters.append(U_list)
        # Save unitaries to file
        if save:
            with open(unitaries_file_name, 'wb') as f:
                pickle.dump(self.U_list_filters, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def generate_qc(self, gates_name='G3', num_gates=300, num_filters=3, num_features=19, save=True,
                    saved_gates_filename='gates_list2D.pickle', saved_qubits_filename='qubits_list2D.pickle'):
        '''
        Generate sets of random quantum gates and their associated qubits and saves them. This function is only used with Qiskit backends.
            gates_name (str): name of the family of quantum gates. Either G1, G2, G3 or Ising
            num_gates (int): depth of the quantum circuits
            num_filters (int): Number of quantum filters to apply to each feature
            num_features (int): Number of features of the data
            save (bool): Save the gates and qubits to pickle files
            pickle_gates_filename (str): File name for gates set
            pickle_qubits_filename (str): File name for qubit set
        '''
        if self.backend == 'torch':
            raise ValueError("This function is only callable with the Qiskit backends")

        self.gates_set_list = []
        self.qubits_set_list = [] 
        # Store circuit parameters
        self.gates_name = gates_name
        self.num_gates = num_gates
        self.num_filters = num_filters
        self.num_features = num_features

        for i in range(num_filters): # Random quantum circuit for each  filter
            gates_list = []
            qubits_list = []
            for j in range(num_features): # Random quantum circuit for each  feature
                gates_set, qubits_set = self._select_gates_qubits(gates_name, num_gates)
                gates_list.append(gates_set)
                qubits_list.append(qubits_set)
            
            self.gates_set_list.append(gates_list)
            self.qubits_set_list.append(qubits_list)

        # Save gates and qubits to file
        if save:
            with open(saved_gates_filename, 'wb') as f:
                pickle.dump(self.gates_set_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(saved_qubits_filename, 'wb') as f:
                pickle.dump(self.qubits_set_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                
    def load_gates(self, gates_name='G3', name_gates='gates_list.pickle', name_qubits='qubits_list.pickle'):
        '''
        Load set of quantum gates and qubits. This function is only used with Qiskit backends.
            name_gates (str): File name for gates set
            name_qubits (str): File name for qubit set
        '''
        if self.backend=='torch':
            raise ValueError("This function is only callable whith the Qiskit backends")
            
        with open(name_gates, 'rb') as f:
            self.gates_set_list = pickle.load(f)    
        with open(name_qubits, 'rb') as f:
            self.qubits_set_list = pickle.load(f) 

        # Store circuit parameters
        self.num_filters = len(self.gates_set_list)
        self.num_features = len(self.gates_set_list[0])
        self.num_gates = len(self.gates_set_list[0][0])
        self.gates_name = gates_name
        nqbits = max(max(max(max(self.qubits_set_list))))+1
        if nqbits!=self.nqbits:
            raise ValueError('Incorrect number of qubits of the loaded quantum circuits')
            
    def load_unitaries(self, name):
        '''
        Loads the unitaries.  This function is only used with Pytorch backend.
            name (str): File name for unitaries
        '''
        if self.backend!='torch':
            raise ValueError("This function is only callable whith the 'torch' backend")
            
        with open(name, 'rb') as f:
            self.U_list_filters = pickle.load(f)
            
        # Store circuit parameters 
        self.num_filters = len(self.U_list_filters)
        self.num_features = len(self.U_list_filters[0])
        nqbits = int(np.ceil(np.log2(self.U_list_filters[0][0].shape[0])))     
        if nqbits!=self.nqbits:
            raise ValueError('Incorrect number of qubits of the loaded quantum circuits')
                
    def _run(self, data, tol=1e-6, n_filt=0):
        """
        Runs the quantum filters for all the features
            data (tensor): input data (one feature), shape (num_samples, num_features, N,N,N)
            tol (float): tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.
            n_filt (int): index of the number of filters
        returns:
            (tensor): output quantum filter
        """
        # Scale data
        data_scaled = self._scale_data(data)
        self.num_samples = data.shape[0] # Store number of samples
        fin_shape = int(data.shape[-1]/self.stride) # Final shape of the data
        if self.backend=='torch':
            data_out = torch.zeros((data.shape[0], data.shape[1],fin_shape,fin_shape))
            for i in range(data.shape[1]): # Run for every feature
                data_out[:,i,:,:] = self._run_filter(data_scaled[:,i,:,:],
                                  tol, torch.tensor(self.U_list_filters[n_filt][i]).to(self.device))
        else:
            data_out = np.zeros((data.shape[0], data.shape[1],fin_shape,fin_shape))
            for i in range(data.shape[0]):    
                for j in range(data.shape[1]):
                    data_out[i,j,:,:] = self._run_filterQiskit(data[i,j,:,:], self.gates_set_list[n_filt][j], self.qubits_set_list[n_filt][j], tol)
            
        return data_out
    
    def get_quantum_filters(self, data,tol=1e-6):   
        """
        Runs the quantum filters for all features multiple times (num_filters times)
            data (tensor): input data (one feature), shape (num_samples, num_features,N,N,N)
            tol (float): tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.
        returns:
            (tensor/np.array): output quantum filters
        """
        if len(data.shape)!=4:
            raise ValueError('Incorrect data shape. The data should have shape (num_samples, num_features, N,N)')
        all_results = [
            self._run(data, tol, n_filt=i)
            for i in range(self.num_filters) # Run for each number of filters
        ]

        # Reshape final array
        if self.backend=='torch':
            results_reshape = torch.zeros([
                all_results[0].shape[0],
                all_results[0].shape[1]*self.num_filters,
                all_results[0].shape[2],all_results[0].shape[3]
            ])
            for i in range(self.num_filters):
                results_reshape[
                    :,
                    ((all_results[0].shape[1])*i):(all_results[0].shape[1])*(i + 1),
                    :,
                    :
                ] = all_results[i]
    
        else:
            all_results = np.array(all_results)
            results_reshape = all_results.reshape(
                all_results[0].shape[0],
                all_results[0].shape[1]*self.num_filters,
                all_results[0].shape[2],
                all_results[0].shape[3]
            )

        return results_reshape


class QuantumFilters3D(QuantumFiltersBase):
    
    description = "Applies a quantum filter to a 3D volume.  Quantum data encoding: Flexible Representation of Quantum Images. Quantum transformation: quantum reservoirs with fixed number of gates. Implemented: G1 = {CNOT, H, X}, G2 = {CNOT, H, S}, and G3={CNOT,H,T} and evolution under the transverse field Ising model. The code is designed to run either in a Qiskit backend or with Pytorch."
    class_parameters = {
        "shape": "(n,n,n), n integer. Box size in which the data is split to apply the quantum filter. The hilbert space is better exploit if we set n^3=2^l.",
        "stride": "int. Stride used to move across the image.",
        "nqbits": "Number of qubits of the quantum circuits. nqbits = [log2(n^3)] + 1. This parameter can not be chosen by the user.",
        "num_filters": "Number of quantum features applied to each of the features of the data. If the input data has shape (n_samples,num_features,N,N,N), the output data has shape (n_samples, num_features*num_filters,N,N,N)",
        "num_samples": "Number of samples of the input data",
        "device": "Pytorch device (cpu, cuda). Only used if the backend is torch",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_set": "List of quantum gates that form the quantum reservoir",
        "qubits_set":"List of qubits to which the quantum gates are applied to",
        "U_list_filters": "list of all the quantum unitaries that define the quantum reservoirs. The list has shape (num_filters, num_features, nqbits, nqbits). Only used if backend is torch.",
        "shape_windows": "shape of the sliding windows",
        "shots": "Number of shots per experiment (only Qiskit backends)",
        "backend": "Backend where to run the code. Either Qiskit bachend or torch"
            }
    required_parameters = {}
    optional_parameters = {
        "shape": "(n,n,n), n integer. Box size in which the data is split to apply the quantum filter. The hilbert space is better exploit if we set n^3=2^l.",  
        "stride": "int. Stride used to move across the image.",
        "shots": "Number of shots per experiment (only Qiskit backends)",
        "backend": "Backend where to run the code. Either Qiskit bachend or torch"
    }
    example_parameters = {
        "shape": (2,2,2),
        "stride": 2,
        "shots": 4096,
        "backend":"aer_simulator"
    }
    
    def __init__(self, shape: tuple=(4,4,4), stride:float=1, shots=4096, backend='torch'):

        super().__init__(n_dimensions=3, shape=shape)

        self.shape = shape
        self.stride = stride
        self.num_filters = 1
        self.shots=shots
        self.backend = backend
        if self.backend == 'torch':
            # set CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(int("cuda:0".split(':')[1]))
            else:
                self.device = torch.device("cpu")
            
          
    def _FRQI_encoding(self, box):
        '''
        Flexible representation of Quantum Images encoding. Takes an (nxnxn) box and returns the encoded quantum circuit. 
        This function is only used with Qiskit backends.
            box (np.array): (nxnxn) box to be encoded
        '''
        # check if the box has the correct shape
        assert box.shape==self.shape

        # Calculate binary string of basis qubits
        qbit_list = list(itertools.product([0, 1], repeat=self.nqbits-1))
        coefs = {}
        l=0
        for i in range(box.shape[0]):
            for j in range(box.shape[1]):
                for k in range(box.shape[2]):
                    theta = box[i,j,k]
                    q_str = ''.join([str(q) for q in qbit_list[l]])
                    coefs[q_str + '0'] = np.cos(theta)/np.sqrt(self.shape[0]**3)
                    coefs[q_str + '1'] = np.sin(theta)/np.sqrt(self.shape[0]**3)
                    l+=1
        # store them in a dictionary and sort it by keys
        coefs = list(collections.OrderedDict(sorted(coefs.items())).values())

        # Define quantum circuit
        cr = ClassicalRegister(self.nqbits-1)
        qr = QuantumRegister(self.nqbits)
        qc = QuantumCircuit(qr,cr)

        # Calculate initial state
        initial_state = np.zeros(2**self.nqbits)
        coefs =  coefs/np.sqrt(np.sum(np.square(coefs)))
        initial_state[:coefs.shape[0]] = coefs
        qc.initialize(initial_state, list(range(self.nqbits)))
        return qc, initial_state    
    
    def _rollNumpy(self, a, b, dx=1, dy=1, dz=1):   
        '''
        Rolling 3D window for numpy array. This function is only used with Qiskit backends.
            a (np.array): input array
            b (np.array): rolling 2D window array
            dx (int): horizontal step, abscissa, number of columns
            dy (int): vertical step, ordinate, number of rows
            dz (int): transverse step, applicate, number of layers
        '''
        shape = a.shape[:-3] + \
                ((a.shape[-3] - b.shape[-3]) // dz + 1,) + \
                ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
                ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
                b.shape  # multidimensional "sausage" with 3D cross-section
        strides = a.strides[:-3] + \
                  (a.strides[-3] * dz,) + \
                  (a.strides[-2] * dy,) + \
                  (a.strides[-1] * dx,) + \
                  a.strides[-3:]
        self.shape_windows = shape
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def _roll(self, a, b, dx=1, dy=1, dz=1):
        '''
        Rolling 3D window for pytorch tensor. This function is only used with Pytorch backends.
            a (tensor): input array, shape (n_samples, N,N,N)
            b (tensor): rolling 3D window array, shape (n,n,n)
            dx (int): horizontal step, abscissa, number of columns
            dy (int): vertical step, ordinate, number of rows
            dz (int): transverse step, applicate, number of layers
        '''
        shape = a.shape[:-3] + \
                ((a.shape[-3] - b.shape[-3]) // dz + 1,) + \
                ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
                ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
                b.shape  # multidimensional "sausage" with 3D cross-section
        strides = a.stride()[:-3] + \
              (a.stride()[-3] * dz,) + \
              (a.stride()[-2] * dy,) + \
              (a.stride()[-1] * dx,) + \
              a.stride()[-3:]

        self.shape_windows = shape
        return torch.as_strided(a, shape, strides)
    
    def _scale_data(self, data):
        """
            Scale the data to [0, pi/2) (each feature is scaled separately)
                data (tensor): input data, shape (n_samples, num_features, N,N,N)
            returns:
                (tensor): scaled data, shape (n_samples, num_features, N,N,N)
        """
        new_min, new_max = 0, np.pi/2
        for i in range(data.shape[0]):# Scale each feature independently
            for j in range(data.shape[1]):
                v_min, v_max = data[i,j,:,:,:].min(), data[i,j,:,:,:].max()
                if v_min < v_max:
                    data[i,j,:,:,:] = (data[i,j,:,:,:] - v_min)/(v_max - v_min)*(new_max - new_min) + new_min                    
        return data
    
    def _run_boxQiskit(self, box, gates_set, qubits_set):
        '''
        For a given box of the image runs the FRQI encoding + random quantum circuit.  This function is only used with Qiskit backends.
            box (np.array): input data
            gates_set (list): Set of random quantum gates
            qubits_set (list): List of qubits to apply the quantum gates
        '''
        # FRQI encoding
        qc, initial_state = self._FRQI_encoding(box)
        
        # Random quantum circuit
        if self.gates_name=='Ising':
            qc_Ising = self._apply_ising_gates()
            qc+=qc_Ising
        else:
            self._apply_G_gates(qc, gates_set, qubits_set, measure=True)
        
        # Get counts
        if self.backend=='aer_simulator':
            aer_sim = Aer.get_backend(self.backend)
            t_qc = transpile(qc, aer_sim)
            qobj = assemble(t_qc, shots=self.shots)
            result = aer_sim.run(qobj).result()
        else: # For real hardware/fake simulators
            coupling_map = self.backend.configuration().coupling_map
            optimized_3 = transpile(qc, backend=self.backend, seed_transpiler=11)
            result = self.backend.run(optimized_3, shots=self.shots).result()
        counts = result.get_counts(qc)
        
        # Calculate binary string of basis qubits
        qbit_list = list(itertools.product([0, 1], repeat=self.nqbits-1))
        new_counts = []
        for l in range(len(qbit_list)):
            q_str =''.join([str(value) for value in qbit_list[l]])
            new_counts.append(counts.get(q_str, 0))

        new_counts/=np.sum(new_counts)
        return new_counts
    
    def _run_filterQiskit(self, data, gates_set, qubits_set, tol=1e-6):
        '''
        Runs the quantum circuit for one feature of the data.  This function is only used with Qiskit backends.
            data (np.array): shape (N,N,N), channel of the image 
            gates_set (list): Set of random quantum gates
            qubits_set (list): List of qubits to apply the quantum gates
            scale (bool): Scale the input image
        '''
        # Get boxes from data
        size = self.shape[0]
        b=np.zeros(self.shape)
        windows = self._rollNumpy(data,b,dx=size*self.stride,dy=size*self.stride,dz=size*self.stride).reshape(-1,size,size,size)       
        
        results = []
        for i in range(windows.shape[0]): # Loop through every box of the image
            # If all values are zero, no need to run the algo
            box = windows[i]
            if np.sum(np.abs(box))>tol: # Run the QC if the box is non-zero valued
                result = np.array(self._run_boxQiskit(box,  gates_set, qubits_set))
                result = result[:self.shape[0]**3].reshape(self.shape)
            else: # Return zeros if the box is all zero-valued
                result = np.array([0 for i in range(self.shape[0]**3)]).reshape(self.shape)
            results.append(result)
        
        # Reshape the results in the original shape
        results = np.array(results).reshape(self.shape_windows) # Reshape results to rolling windows shape
        fin_shape = int(data.shape[-1]/self.stride)
        results = results.transpose([0, 3, 1, 4, 2, 5]).reshape(fin_shape,fin_shape,fin_shape)# Transpose to obtain original shape
        
        # Apply zero-mask
        mask = np.ones(self.shape_windows) 
        windows_reshape = windows.reshape(self.shape_windows)# Reshape to rolling windows shape
        mask[np.abs(windows_reshape)<tol] = 0 #Find zero-valued values of the original data
        mask = mask.reshape(self.shape_windows).transpose([0, 3, 1, 4, 2, 5]).reshape((fin_shape,fin_shape,fin_shape))# Transpose to return to original shape
        results = results.astype('float64')
        results*=mask #Apply mask
        return results
    
    def _run_filter(self, data,tol, U):
        """
            Runs a quantum filter to a feature from the data samples.  This function is only used with Pytorch backend.
                data (tensor): input data (one feature), shape (num_samples, N,N,N)
                tol (float): tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.
                U (tensor): Unitary matrix representing the random quantum circuit
            returns:
                (tensor): quantum filter applied to data
        """
        # 0. Get rolling windows
        size = self.shape[0]
        b=torch.zeros(self.shape)
        windows = self._roll(data,b,dx=size*self.stride,dy=size*self.stride,dz=size*self.stride).reshape(-1,size,size,size).to(self.device)


        # 1. Flexible representation of quantum images. Apply cos(theta)|0> + sin(theta)|1> transformation
        v1 = torch.tensor([1,0]).to(self.device)
        v2 = torch.tensor([0,1]).to(self.device)
        sq = torch.tensor([self.shape[0]**3]).to(self.device)

        frqi = (torch.kron(torch.cos(windows),  v1) +
           torch.kron(torch.sin(windows) , v2))/torch.sqrt(sq)

        # 2. Flatten the last dimension to create an initial state
        flatten = frqi.reshape(-1,2**(self.nqbits)).type(torch.complex128)
        # 3. Apply U rotation to each state
        apply_U = (U @ flatten.T).reshape(-1,size, size, size, 2)

        # 4. Partial trace to remove the extra qubit
        partial_trace = torch.einsum('ijklm->ijkl', apply_U)

        # 5. Reshape back to window view
        partial_trace_reshaped = partial_trace.reshape(self.shape_windows)

        # 6. Reshape to original shape, taking into account transposition
        fin_shape = int(data.shape[-1]/self.stride)
        output = partial_trace_reshaped.permute([0, 1, 4, 2, 5, 3, 6]).reshape((self.num_samples, 
                                                                 fin_shape,fin_shape,fin_shape))

        # 7. Get the probability for each state (modulus of the final state)
        output_probability = (output.conj()*output).real

        # 8. Create a mask to map to zero all zero values of the original image
        mask = torch.ones(windows.shape)
        mask[np.abs(windows)<tol] = 0
        mask = mask.reshape(self.shape_windows).permute([0, 1, 4, 2, 5, 3, 6]).reshape((self.num_samples,
                                                                                      fin_shape,fin_shape,fin_shape)).to(self.device)
        quantum_filter = output_probability*mask        
        return quantum_filter
        
    
    def generate_unitaries(self, gates_name='G3',num_gates=300, num_filters=3, num_features=19, unitaries_file_name='unitaries.pickle', save=True):
        """
        Generates the quantum unitaries that represent the quantum reservoirs (random quantum circuits).  This function is only used with Pytorch backend.
            gates_name (str): name of the family of quantum gates. Either G1, G2, G3 or Ising
            num_gates (int): depth of the quantum circuits
            num_filters (int): Number of quantum filters to apply to each feature
            num_features (int): Number of features of the data
            unitaries_file_name (str): name of the file containing unitary list (only needed if save=True)
            save (bool): if True, the generated unitaries are saved to a pickle file.
        """
        if self.backend != 'torch':
            raise ValueError("This function is only callable with the 'torch' backend")
        self.U_list_filters = []
        self.num_filters = num_filters
        self.gates_name = gates_name
        for i in range(num_filters):
            U_list = []
            for j in range(num_features):
                # Store circuit parameters
                self.gates_name = gates_name
                self.num_gates = num_gates
                # If the quantum reservoir is the Ising model
                if gates_name=='Ising':
                    qc = self._apply_ising_gates()
                else: # Otherwise, if it is one of the G families
                    gates_set, qubits_set = self._select_gates_qubits(gates_name, num_gates)
                    # Get unitary
                    qc = QuantumCircuit(self.nqbits)
                    # Random quantum circuit
                    self._apply_G_gates(qc, gates_set, qubits_set)
                # Get unitary
                backend = BasicAer.get_backend('unitary_simulator')
                job = backend.run(transpile(qc, backend))
                U = job.result().get_unitary(qc)
                U_list.append(U)
            self.U_list_filters.append(U_list)
        # Save unitaries to file
        if save:
            with open(unitaries_file_name, 'wb') as f:
                pickle.dump(self.U_list_filters, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def generate_qc(self, gates_name='G3', num_gates=300, num_filters=3, num_features=19, save=True,
                    name_gates='gates_list.pickle', name_qubits = 'qubits_list.pickle'):
        '''
        Generate sets of random quantum gates and their associated qubits and saves them. This function is only used with Qiskit backends.
            gates_name (str): name of the family of quantum gates. Either G1, G2, G3 or Ising
            num_gates (int): depth of the quantum circuits
            num_filters (int): Number of quantum filters to apply to each feature
            num_features (int): Number of features of the data
            save (bool): Save the gates and qubits to pickle files
            name_gates (str): File name for gates set
            name_qubits (str): File name for qubit set
        '''
        self.gates_set_list = []
        self.qubits_set_list = [] 
        # Store circuit parameters
        self.gates_name = gates_name
        self.num_gates = num_gates
        self.num_filters = num_filters
        self.num_features = num_features
        for i in range(num_filters): # Random quantum circuit for each  filter
            gates_list = []
            qubits_list = []
            for j in range(num_features): # Random quantum circuit for each  feature
                gates_set, qubits_set = self._select_gates_qubits(gates_name, num_gates)
                gates_list.append(gates_set)
                qubits_list.append(qubits_set)
            
            self.gates_set_list.append(gates_list)
            self.qubits_set_list.append(qubits_list)
        # Save gates and qubits to file
        if save:
            with open(name_gates, 'wb') as f:
                pickle.dump(self.gates_set_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(name_qubits, 'wb') as f:
                pickle.dump(self.qubits_set_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                
    def load_gates(self, gates_name ='G3',name_gates='gates_list.pickle', name_qubits = 'qubits_list.pickle'):
        '''
        Load set of quantum gates and qubits. This function is only used with Qiskit backends.
            name_gates (str): File name for gates set
            name_qubits (str): File name for qubit set
        '''
        with open(name_gates, 'rb') as f:
            self.gates_set_list = pickle.load(f)    

        with open(name_qubits, 'rb') as f:
            self.qubits_set_list = pickle.load(f) 
        # Sotre circuit parameters
        self.num_filters = len(self.gates_set_list)
        self.num_features = len(self.gates_set_list[0])
        self.num_gates = len(self.gates_set_list[0][0])
        self.gates_name = gates_name
        nqbits = max(max(max(max(self.qubits_set_list))))+1
        if nqbits!=self.nqbits:
            raise ValueError('Incorrect number of qubits of the loaded quantum circuits')
            
    def load_unitaries(self, name):
        '''
        Loads the unitaries.  This function is only used with Pytorch backend.
            name (str): File name for unitaries
        '''
        with open(name, 'rb') as f:
            self.U_list_filters = pickle.load(f)
        # Store circuit parameters 
        self.num_filters = len(self.U_list_filters)
        self.num_features = len(self.U_list_filters[0])
        nqbits = int(np.ceil(np.log2(self.U_list_filters[0][0].shape[0])))     
        if nqbits!=self.nqbits:
            raise ValueError('Incorrect number of qubits of the loaded quantum circuits', nqbits, self.nqbits)
                
                
    def _run(self, data, tol=1e-6, n_filt=0):
        """
        Runs the quantum filters for all the features
            data (tensor): input data (one feature), shape (num_samples, num_features, N,N,N)
            tol (float): tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.
            n_filt (int): index of the number of filters
        returns:
            (tensor): output quantum filter
        """
        # Scale data
        data_scaled = self._scale_data(data)
        self.num_samples = data.shape[0] # Store number of samples
        fin_shape = int(data.shape[-1]/self.stride) # Final shape of the data
        if self.backend=='torch':
            data_out = torch.zeros((data.shape[0], data.shape[1],fin_shape,fin_shape,fin_shape))
            for i in range(data.shape[1]): # Run for every feature
                data_out[:,i,:,:,:] = self._run_filter(data_scaled[:,i,:,:,:],
                                  tol, torch.tensor(self.U_list_filters[n_filt][i]).to(self.device))
        else:
            data_out = np.zeros((data.shape[0], data.shape[1],fin_shape,fin_shape,fin_shape))
            for i in range(data.shape[0]):    
                for j in range(data.shape[1]):
                    data_out[i,j,:,:,:] = self._run_filterQiskit(data[i,j,:,:,:], self.gates_set_list[n_filt][j], self.qubits_set_list[n_filt][j], tol)
            
        return data_out
    
    def get_quantum_filters(self, data,tol=1e-6):   
        """
        Runs the quantum filters for all features multiple times (num_filters times)
            data (tensor): input data (one feature), shape (num_samples, num_features,N,N,N)
            tol (float): tolerance for the masking matrix. All values from the original data which are smaller than the tolerance are set to 0.
        returns:
            (tensor/np.array): output quantum filters
        """
        if len(data.shape)!=5:
            raise ValueError('Incorrect data shape. The data should have shape (num_samples, num_features, N,N,N)')
        all_results = []
        for i in range(self.num_filters): # Run for each number of filters
            all_results.append(self._run( data, tol, n_filt=i))
                
        # Reshape final array
        if self.backend=='torch':
            results_reshape = torch.zeros([all_results[0].shape[0], all_results[0].shape[1]*len(all_results),
                               all_results[0].shape[2],all_results[0].shape[3],all_results[0].shape[4] ])
            for i in range(1,len(all_results)+1):
                results_reshape[:,((all_results[0].shape[1])*(i-1)):(all_results[0].shape[1])*i,:,:,:] = all_results[i-1]
    
        else:
            all_results = np.array(all_results)
            results_reshape = all_results.reshape(all_results.shape[1], self.num_filters*all_results.shape[2],
                                              all_results.shape[3],all_results.shape[3],all_results.shape[3])      
        return results_reshape