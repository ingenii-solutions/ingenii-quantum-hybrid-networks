import collections
import itertools

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit import BasicAer, Aer, assemble, transpile
from qiskit.quantum_info import Pauli
from qiskit import opflow

import numpy as np
import pickle
from random import sample
import torch

from .utils import roll_numpy, roll_torch


class QuantumFiltersBase():
    def __init__(
            self, n_dimensions: int, shape: tuple, stride: float,
            shots: int, backend: str):

        self.n_dimensions = n_dimensions

        # Calculate the number of qubits
        self.nqbits = int(np.ceil(np.log2(shape[0]**n_dimensions)) + 1)

        self.shape = shape
        self.stride = stride
        self.shots = shots

        self.backend = backend
        if self.backend == 'torch':
            # set CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(int("cuda:0".split(':')[1]))
            else:
                self.device = torch.device("cpu")

        # Initialise
        self.unitaries_list = []
        self.num_filters = 0
        self.num_features = 0

    gate_names = {
        "G1": ["CNOT", "H", "X"],
        "G2": ["CNOT", "H", "S"],
        "G3": ["CNOT", "H", "T"],
    }

    def _select_gates_qubits(self, gates_name, num_gates):
        """
        Selects random gates from the G1, G2 and G3 family, and the qubits to
            which the gates are applied to.
        gates_name (str): name of the family of quantum gates. Either G1, G2,
            G3 or Ising
        num_gates (int): depth of the quantum circuits
        """
        # Select random gate
        gates = self.gate_names[gates_name]
        gates_set = [sample(gates, 1)[0] for _ in range(num_gates)]

        qubit_idx = list(range(self.nqbits))

        def _add_qubit_set(gate):
            if gate == 'CNOT':
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
        Selects a set of random qubits and random gates from the G1, G2 or G3
        family, and stores them in a list.
        These gates form the random quantum circuit.
            qc (QuantumCircuit): quantum circuit
            gates_set (list): List of quantum gates that form the quantum
                reservoir
            qubits_set (list): List of qubits to which the quantum gates are
                applied to
            measure (bool): mneasure all the qubits (except the ancilla) at the
                end of the circuit
        """
        # Apply random gates to random qubits
        for gate, qubit in zip(gates_set, qubits_set):

            if gate == 'CNOT':  # For 2-qubit gates
                # Select qubit 1 and 2 (different qubits)
                qbit1, qbit2 = qubit
                # Apply gate to qubits
                qc.cx(qbit1, qbit2)

            else:  # For 1-qubit gates
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

    def _apply_ising_gates(self, t: float = 10):
        """
        Applies quantum evolution under the transverse field Ising model.
        The parameters of the Ising model are chosen according to
        R. Martínez-Peña et al. (2021).
        Js ~ N(0.75, 0.1), Jij ~ U(-Js/2, Js/2) and hi = h such that h/Js = 0.1
            t (float): time evolution
        returns:
            (QuantumCircuit): quantum circuit representing the evolution of
            the Ising model
        """
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
        for q1_idx, q2_idx in itertools.combinations(qubit_idxs, 2):
            # Interaction operators
            name = \
                name_gate[:q1_idx] + 'Z' + \
                name_gate[q1_idx+1:q2_idx] + \
                'Z' + name_gate[q2_idx+1:]
            coef = np.random.uniform(-Js/2, Js/2)
            pauli_op += coef*opflow.PauliOp(Pauli(name))

        for q_idx in qubit_idxs:  # Single qubit operators
            name = name_gate[:q_idx] + 'X' + name_gate[(q_idx+1):]
            coef = h
            pauli_op += coef*opflow.PauliOp(Pauli(name))

        # Time evolution operator exp(iHt)
        evo_time = Parameter('θ')
        evolution_op = (evo_time*pauli_op).exp_i()

        # Trotterization
        trotterized_op = opflow.PauliTrotterEvolution(
            trotter_mode=opflow.Suzuki(order=2, reps=1)).convert(evolution_op)
        bound = trotterized_op.bind_parameters({evo_time: t})

        # Convert the trotterized operator to a quantum circuit
        qc_ham = bound.to_circuit()

        return qc_ham

    def generate_unitaries(
            self, gates_name='G3', num_gates=300, num_filters=3,
            num_features=19, unitaries_file_name='unitaries.pickle',
            save=True):
        """
        Generates the quantum unitaries that represent the quantum reservoirs
        (random quantum circuits)
        This function is only used with Pytorch backend
            gates_name (str): name of the family of quantum gates. Either G1,
            G2, G3 or Ising
            num_gates (int): depth of the quantum circuits
            num_filters (int): Number of filters to apply to each feature
            num_features (int): Number of features of the data
            unitaries_file_name (str): name of the file containing unitary list
                (only needed if save=True)
            save (bool): Whether the generated unitaries are saved to a file
        """
        if self.backend != 'torch':
            raise ValueError(
                "This function is only callable with the 'torch' backend")

        self.unitaries_list = []
        self.num_filters = num_filters
        self.gates_name = gates_name
        for i in range(num_filters):
            U_list = []
            for j in range(num_features):
                # Store circuit parameters
                self.num_gates = num_gates
                if gates_name == 'Ising':
                    # If the quantum reservoir is the Ising model
                    qc = self._apply_ising_gates()
                else:
                    # Otherwise, if it is one of the G families
                    gates_set, qubits_set = \
                        self._select_gates_qubits(gates_name, num_gates)
                    # Get unitary
                    qc = QuantumCircuit(self.nqbits)
                    # Random quantum circuit
                    self._apply_G_gates(qc, gates_set, qubits_set)
                # Get unitary
                backend = BasicAer.get_backend('unitary_simulator')
                job = backend.run(transpile(qc, backend))
                U = job.result().get_unitary(qc)
                U_list.append(U)
            self.unitaries_list.append(U_list)
        # Save unitaries to file
        if save:
            with open(unitaries_file_name, 'wb') as f:
                pickle.dump(
                    self.unitaries_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _check_file_names(self, saved_gates_filename, saved_qubits_filename):

        saved_gates_filename = \
            saved_gates_filename or f'gates_list_{self.n_dimensions}D.pickle'
        saved_qubits_filename = \
            saved_qubits_filename or f'qubits_list_{self.n_dimensions}D.pickle'

        return saved_gates_filename, saved_qubits_filename

    def generate_qc(
            self, gates_name='G3', num_gates=300, num_filters=3,
            num_features=19, save=True,
            saved_gates_filename=None, saved_qubits_filename=None):
        '''
        Generate sets of random quantum gates and their associated qubits and
        saves them. This function is only used with Qiskit backends.
            gates_name (str): name of the family of quantum gates. Either G1,
            G2, G3 or Ising
            num_gates (int): depth of the quantum circuits
            num_filters (int): Number of filters to apply to each feature
            num_features (int): Number of features of the data
            save (bool): Save the gates and qubits to pickle files
            saved_gates_filename (str): File name for saved gates set
            saved_qubits_filename (str): File name for saved qubit set
        '''
        if self.backend == 'torch':
            raise ValueError(
                "This function is only callable with the Qiskit backends")

        saved_gates_filename, saved_qubits_filename = \
            self._check_file_names(saved_gates_filename, saved_qubits_filename)

        self.gates_set_list = []
        self.qubits_set_list = []
        # Store circuit parameters
        self.gates_name = gates_name
        self.num_gates = num_gates
        self.num_filters = num_filters
        self.num_features = num_features

        for _ in range(num_filters):
            # Random quantum circuit for each filter
            gates_list = []
            qubits_list = []
            for _ in range(num_features):
                # Random quantum circuit for each feature
                gates_set, qubits_set = \
                    self._select_gates_qubits(gates_name, num_gates)
                gates_list.append(gates_set)
                qubits_list.append(qubits_set)

            self.gates_set_list.append(gates_list)
            self.qubits_set_list.append(qubits_list)

        # Save gates and qubits to file
        if save:
            with open(saved_gates_filename, 'wb') as f:
                pickle.dump(
                    self.gates_set_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(saved_qubits_filename, 'wb') as f:
                pickle.dump(
                    self.qubits_set_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_gates(
            self, gates_name='G3',
            saved_gates_filename=None, saved_qubits_filename=None):
        '''
        Load set of quantum gates and qubits. This function is only used with
        Qiskit backends.
            gates_name (str): name of the family of quantum gates. Either G1,
            G2, G3 or Ising
            saved_gates_filename (str): File name for saved gates set
            saved_qubits_filename (str): File name for saved qubit set
        '''

        saved_gates_filename, saved_qubits_filename = \
            self._check_file_names(saved_gates_filename, saved_qubits_filename)

        if self.backend == 'torch':
            raise ValueError(
                "This function is only callable whith the Qiskit backends")

        with open(saved_gates_filename, 'rb') as f:
            self.gates_set_list = pickle.load(f)
        with open(saved_qubits_filename, 'rb') as f:
            self.qubits_set_list = pickle.load(f)

        # Store circuit parameters
        self.num_filters = len(self.gates_set_list)
        self.num_features = len(self.gates_set_list[0])
        self.num_gates = len(self.gates_set_list[0][0])
        self.gates_name = gates_name
        nqbits = max(max(max(max(self.qubits_set_list)))) + 1

        if nqbits != self.nqbits:
            raise ValueError(
                'Incorrect number of qubits of the loaded quantum circuits')

    def load_unitaries(self, file_name):
        '''
        Loads the unitaries.  This function is only used with Pytorch backend.
            file_name (str): File name for unitaries
        '''
        if self.backend != 'torch':
            raise ValueError(
                "This function is only callable with the 'torch' backend")

        with open(file_name, 'rb') as f:
            self.unitaries_list = pickle.load(f)

        # Store circuit parameters
        self.num_filters = len(self.unitaries_list)
        self.num_features = len(self.unitaries_list[0])
        nqbits = int(np.ceil(np.log2(self.unitaries_list[0][0].shape[0])))

        if nqbits != self.nqbits:
            raise ValueError(
                'Incorrect number of qubits of the loaded quantum circuits')

    def _FRQI_encoding(self, box):
        '''
        Flexible representation of Quantum Images encoding. Takes an (nxn) or
        (nxnxn) box and returns the encoded quantum circuit.
        This function is only used with Qiskit backends.
            box (np.array): (nxn) or (nxnxn) box to be encoded
        '''
        # check if the box has the correct shape
        assert box.shape == self.shape

        # Calculate binary string of basis qubits
        qbit_list = list(itertools.product([0, 1], repeat=self.nqbits-1))
        coefs = {}

        def _add_coeffs(theta, idx):
            q_str = ''.join([str(q) for q in qbit_list[idx]])
            coefs[q_str + '0'] = np.cos(theta) / \
                np.sqrt(self.shape[0]**self.n_dimensions)
            coefs[q_str + '1'] = np.sin(theta) / \
                np.sqrt(self.shape[0]**self.n_dimensions)

        l_idx = 0
        for i in range(box.shape[0]):
            for j in range(box.shape[1]):
                if self.n_dimensions == 2:
                    _add_coeffs(box[i, j], l_idx)
                    l_idx += 1
                elif self.n_dimensions == 3:
                    for k in range(box.shape[2]):
                        _add_coeffs(box[i, j, k], l_idx)
                        l_idx += 1

        # store them in a dictionary and sort it by keys
        coefs = list(collections.OrderedDict(sorted(coefs.items())).values())

        # Define quantum circuit
        cr = ClassicalRegister(self.nqbits-1)
        qr = QuantumRegister(self.nqbits)
        qc = QuantumCircuit(qr, cr)

        # Calculate initial state
        initial_state = np.zeros(2**self.nqbits)
        coefs = coefs/np.sqrt(np.sum(np.square(coefs)))
        initial_state[:coefs.shape[0]] = coefs
        qc.initialize(initial_state, list(range(self.nqbits)))
        return qc, initial_state

    def _scale_data(self, data):
        """
        Scale the data to [0, pi/2) (each feature is scaled separately)
            data (tensor): input data, shape (n_samples, num_features, N,N,N)
        returns:
            (tensor): scaled data, shape (n_samples, num_features, N,N,N)
        """
        new_min, new_max = 0, np.pi/2
        for i in range(data.shape[0]):  # Scale each feature independently
            for j in range(data.shape[1]):

                if self.n_dimensions == 2:
                    data_slice = data[i, j, :, :]
                elif self.n_dimensions == 3:
                    data_slice = data[i, j, :, :, :]

                v_min, v_max = data_slice.min(), data_slice.max()
                if v_min < v_max:
                    data_slice = (data_slice - v_min) / \
                        (v_max - v_min)*(new_max - new_min) \
                        + new_min
        return data

    def _run_boxQiskit(self, box, gates_set, qubits_set):
        '''
        For a given box of the image runs the FRQI encoding + random quantum
        circuit.  This function is only used with Qiskit backends.
            box (np.array): shape (n,n). Input data
            gates_set (list): Set of random quantum gates
            qubits_set (list): List of qubits to apply the quantum gates
        '''
        # FRQI encoding
        qc, initial_state = self._FRQI_encoding(box)

        # Random quantum circuit
        if self.gates_name == 'Ising':
            qc_Ising = self._apply_ising_gates()
            qc += qc_Ising
        else:
            self._apply_G_gates(qc, gates_set, qubits_set, measure=True)

        # Get counts
        if self.backend == 'aer_simulator':
            aer_sim = Aer.get_backend(self.backend)
            t_qc = transpile(qc, aer_sim)
            qobj = assemble(t_qc, shots=self.shots)
            result = aer_sim.run(qobj).result()
        else:  # For real hardware/fake simulators
            optimized_3 = transpile(
                qc, backend=self.backend, seed_transpiler=11)
            result = self.backend.run(optimized_3, shots=self.shots).result()
        counts = result.get_counts(qc)

        # Calculate binary string of basis qubits
        new_counts = [
            counts.get(''.join([str(value) for value in qbits]), 0)
            for qbits in itertools.product([0, 1], repeat=self.nqbits-1)
        ]
        new_counts /= np.sum(new_counts)

        return new_counts

    def _run_filter_qiskit(self, data, gates_set, qubits_set, tol=1e-6):
        '''
        Runs the quantum circuit for one feature of the data.  This function
        is only used with Qiskit backends.
            data (np.array): shape (N,N), channel of the image
            gates_set (list): Set of random quantum gates
            qubits_set (list): List of qubits to apply the quantum gates
            scale (bool): Scale the input image
        '''

        size = self.shape[0]

        transpose_idxs = {
            2: [0, 2, 1, 3],
            3: [0, 3, 1, 4, 2, 5]
        }

        # Get boxes from data
        strided_window, shape = roll_numpy(
            data, np.zeros(self.shape),
            dx=size*self.stride, dy=size*self.stride,
            dz=size*self.stride if self.n_dimensions == 3 else None
        )
        self.shape_windows = shape
        windows = strided_window.reshape((-1,) + (size,) * self.n_dimensions)

        def _run_per_box(box):
            if np.sum(np.abs(box)) > tol:
                # Run the QC if the box is non-zero valued
                result = np.array(
                    self._run_boxQiskit(box, gates_set, qubits_set))
                return result[:size**self.n_dimensions].reshape(self.shape)
            else:
                # Return zeros if the box is all zero-valued
                return np.array(
                    [0] * size**self.n_dimensions
                ).reshape(self.shape)

        results = [
            _run_per_box(windows[i])
            for i in range(windows.shape[0])  # Every box of the image
        ]

        fin_shape = int(data.shape[-1]/self.stride)
        fin_shape_parameters = (fin_shape,) * self.n_dimensions

        # Reshape the results in the original shape
        results = np.array(results).reshape(self.shape_windows)
        results = results.transpose(transpose_idxs[self.n_dimensions]) \
            .reshape(fin_shape_parameters)  # Transpose to original shape

        # Apply zero-mask
        mask = np.ones(self.shape_windows)

        # Reshape to rolling windows shape
        windows_reshape = windows.reshape(self.shape_windows)

        # Find zero-valued values of the original data
        mask[np.abs(windows_reshape) < tol] = 0

        mask = mask.reshape(self.shape_windows) \
            .transpose(transpose_idxs[self.n_dimensions]) \
            .reshape(fin_shape_parameters)  # Transpose to original shape
        results = results.astype('float64')
        results *= mask  # Apply mask

        return results

    def _run_filter_torch(self, data, tol, U):
        """
        Runs a quantum filter to a feature from the data samples.
        This function is only used with Pytorch backend.
            data (tensor): input data (one feature), shape (num_samples, N,N,N)
            tol (float): tolerance for the masking matrix. All values from the
            original data which are smaller than the tolerance are set to 0
            U (tensor): Unitary matrix representing the random quantum circuit
        returns:
            (tensor): quantum filter applied to data
        """
        size = self.shape[0]
        reshape_idxs = (-1,) + (size,) * self.n_dimensions

        # Set parameters based on number of dimensions
        if self.n_dimensions == 2:
            einsum_equation = 'ijkl->ijk'
            permutation_idxs = [0, 1, 3, 2, 4]
            dz = None
        elif self.n_dimensions == 3:
            einsum_equation = 'ijklm->ijkl'
            permutation_idxs = [0, 1, 4, 2, 5, 3, 6]
            dz = size*self.stride

        # 0. Get rolling windows
        strided_window, shape = roll_torch(
            data, torch.zeros(self.shape),
            dx=size*self.stride, dy=size*self.stride, dz=dz
        )
        self.shape_windows = shape
        windows = strided_window.reshape(*reshape_idxs).to(self.device)

        # 1. Flexible representation of quantum images.
        # Apply cos(theta)|0> + sin(theta)|1> transformation
        v1 = torch.tensor([1, 0]).to(self.device)
        v2 = torch.tensor([0, 1]).to(self.device)
        sq = torch.tensor([size**self.n_dimensions]).to(self.device)

        frqi = (
            torch.kron(torch.cos(windows), v1) +
            torch.kron(torch.sin(windows), v2)
        ) / torch.sqrt(sq)

        # 2. Flatten the last dimension to create an initial state
        flatten = frqi.reshape(-1, 2**(self.nqbits)).type(torch.complex128)
        # 3. Apply U rotation to each state
        apply_U = (U @ flatten.T).reshape(reshape_idxs + (2,))

        # 4. Partial trace to remove the extra qubit
        partial_trace = torch.einsum(einsum_equation, apply_U)

        # 5. Reshape back to window view
        partial_trace_reshaped = partial_trace.reshape(self.shape_windows)

        # 6. Reshape to original shape, taking into account transposition
        fin_shape = int(data.shape[-1]/self.stride)
        new_reshape_idxs = \
            (self.num_samples,) + (fin_shape,) * self.n_dimensions
        output = partial_trace_reshaped.permute(
            permutation_idxs).reshape(new_reshape_idxs)

        # 7. Get the probability for each state (modulus of the final state)
        output_probability = (output.conj()*output).real

        # 8. Create a mask to map to zero all zero values of the original image
        mask = torch.ones(windows.shape)
        mask[np.abs(windows) < tol] = 0
        mask = mask.reshape(self.shape_windows).permute(permutation_idxs) \
            .reshape(new_reshape_idxs).to(self.device)

        quantum_filter = output_probability*mask
        return quantum_filter

    def _run(self, data, tol=1e-6, n_filt=0):
        """
        Runs the quantum filters for all the features
            data (tensor): input data (one feature), shape
                (num_samples, num_features, N,N,N)
            tol (float): tolerance for the masking matrix. All values from the
                original data which are smaller than the tolerance are set to 0
            n_filt (int): index of the number of filters
        returns:
            (tensor): output quantum filter
        """
        # Scale data
        data_scaled = self._scale_data(data)
        self.num_samples = data.shape[0]  # Store number of samples
        fin_shape = int(data.shape[-1]/self.stride)  # Final shape of the data

        if self.backend == 'torch':
            data_out = torch.zeros(
                (data.shape[0], data.shape[1],) +
                (fin_shape,) * self.n_dimensions
            )

            for i in range(data.shape[1]):  # Run for every feature
                unitary_matrix = torch.tensor(
                    self.unitaries_list[n_filt][i]
                ).to(self.device)

                if self.n_dimensions == 2:
                    data_out[:, i, :, :] = self._run_filter_torch(
                        data_scaled[:, i, :, :], tol, unitary_matrix
                    )
                elif self.n_dimensions == 3:
                    data_out[:, i, :, :, :] = self._run_filter_torch(
                        data_scaled[:, i, :, :, :], tol, unitary_matrix
                    )
        else:
            data_out = np.zeros(
                (data.shape[0], data.shape[1],) +
                (fin_shape,) * self.n_dimensions
            )

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    gates_set = self.gates_set_list[n_filt][j]
                    qubits_set = self.qubits_set_list[n_filt][j]

                    if self.n_dimensions == 2:
                        data_out[i, j, :, :] = self._run_filter_qiskit(
                            data[i, j, :, :], gates_set, qubits_set, tol
                        )
                    elif self.n_dimensions == 3:
                        data_out[i, j, :, :, :] = self._run_filter_qiskit(
                            data[i, j, :, :, :], gates_set, qubits_set, tol
                        )

        return data_out

    def get_quantum_filters(self, data, tol=1e-6):
        """
        Runs the quantum filters for all features multiple (num_filters) times
            data (tensor): input data (one feature),
                shape 2D (num_samples, num_features, N, N),
                3D (num_samples, num_features, N, N, N)
            tol (float): tolerance for the masking matrix. All values from the
                original data which are smaller than the tolerance are set to 0
        returns:
            (tensor/np.array): output quantum filters
        """
        if len(data.shape) != self.n_dimensions + 2:
            raise ValueError(
                'Incorrect data shape. The data should have shape'
                ' (num_samples, num_features,) + (N,) * number of dimensions')
        all_results = [
            self._run(data, tol, n_filt=i)
            for i in range(self.num_filters)  # Run for each number of filters
        ]

        result_shape = all_results[0].shape
        final_result_shape = [
            result_shape[0],
            result_shape[1]*self.num_filters,
        ] + [
            result_shape[2 + i]
            for i in range(self.n_dimensions)
        ]

        # Reshape final array
        if self.backend == 'torch':
            results_reshaped = torch.zeros(final_result_shape)
            for i, result in enumerate(all_results):
                results_reshaped[
                    :,
                    ((result_shape[1])*i):(result_shape[1])*(i + 1),
                    :,
                    :
                ] = result

        else:
            all_results = np.array(all_results)
            results_reshaped = all_results.reshape(final_result_shape)

        return results_reshaped


class QuantumFilters2D(QuantumFiltersBase):

    description = "Applies a quantum filter to a 2D image.  "
    "Quantum data encoding: Flexible Representation of Quantum Images. "
    "Quantum transformation: quantum reservoirs with fixed number of gates. "
    "Implemented: G1 = {CNOT, H, X}, G2 = {CNOT, H, S}, and G3={CNOT,H,T} and "
    "evolution under the transverse field Ising model. The code is designed to"
    " run either in a Qiskit backend or with Pytorch."
    class_parameters = {
        "shape": "(n,n), n integer. Box size in which the data is split to "
        "apply the quantum filter. The hilbert space is better exploit if we "
        "set n^2=2^l.",
        "stride": "int. Stride used to move across the image.",
        "nqbits": "Number of qubits of the quantum circuits. nqbits = "
        "[log2(n^2)] + 1. This parameter can not be chosen by the user.",
        "num_filters": "Number of quantum features applied to each of the "
        "features of the data. If the input data has shape "
        "(n_samples,num_features,N,N), the output data has shape "
        "(n_samples, num_features*num_filters,N,N,N)",
        "num_samples": "Number of samples of the input data",
        "device": "Pytorch device (cpu, cuda). Only used if the backend is "
        "torch",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_set": "List of quantum gates that form the quantum reservoir",
        "qubits_set": "List of qubits to which the quantum gates are applied "
        "to",
        "unitaries_list": "list of all the quantum unitaries that define the"
        " quantum reservoirs. The list has shape (num_filters, num_features, "
        "nqbits, nqbits). Only used if backend is torch.",
        "shape_windows": "shape of the sliding windows",
        "shots": "Number of shots per experiment (only Qiskit backends)",
        "backend": "Backend where to run the code. Either Qiskit or torch"
            }
    required_parameters = {}
    optional_parameters = {
        "shape": "(n,n), n integer. Box size in which the data is split to "
        "apply the quantum filter. The hilbert space is better exploit if we "
        "set n^2=2^l.",
        "stride": "int. Stride used to move across the image.",
        "shots": "Number of shots per experiment (only Qiskit backends)",
        "backend": "Backend where to run the code. Either Qiskit or torch"
    }
    example_parameters = {
        "shape": (2, 2),
        "stride": 2,
        "shots": 4096,
        "backend": "aer_simulator"
    }

    def __init__(
            self, shape: tuple = (4, 4), stride: float = 1,
            shots: int = 4096, backend: str = 'torch'):

        super().__init__(
            n_dimensions=2, shape=shape, stride=stride,
            shots=shots, backend=backend)


class QuantumFilters3D(QuantumFiltersBase):

    description = "Applies a quantum filter to a 3D volume.  "
    "Quantum data encoding: Flexible Representation of Quantum Images. "
    "Quantum transformation: quantum reservoirs with fixed number of gates. "
    "Implemented: G1 = {CNOT, H, X}, G2 = {CNOT, H, S}, and G3={CNOT,H,T} and "
    "evolution under the transverse field Ising model. The code is designed to"
    " run either in a Qiskit backend or with Pytorch."
    class_parameters = {
        "shape": "(n,n,n), n integer. Box size in which the data is split to "
        "apply the quantum filter. The hilbert space is better exploit if we "
        "set n^3=2^l.",
        "stride": "int. Stride used to move across the image.",
        "nqbits": "Number of qubits of the quantum circuits. nqbits = "
        "[log2(n^3)] + 1. This parameter can not be chosen by the user.",
        "num_filters": "Number of quantum features applied to each of the "
        "features of the data. If the input data has shape "
        "(n_samples,num_features,N,N,N), the output data has shape "
        "(n_samples, num_features*num_filters,N,N,N)",
        "num_samples": "Number of samples of the input data",
        "device": "Pytorch device (cpu, cuda). Only used with torch backend",
        "num_gates": "Number of quantum gates of the quantum reservoir",
        "gates_set": "List of quantum gates that form the quantum reservoir",
        "qubits_set": "List of qubits to which the quantum gates are applied"
        " to",
        "unitaries_list": "list of all the quantum unitaries that define the "
        "quantum reservoirs. The list has shape (num_filters, num_features, "
        "nqbits, nqbits). Only used if backend is torch.",
        "shape_windows": "shape of the sliding windows",
        "shots": "Number of shots per experiment (only Qiskit backends)",
        "backend": "Backend where to run the code. Either Qiskit or torch"
            }
    required_parameters = {}
    optional_parameters = {
        "shape": "(n,n,n), n integer. Box size in which the data is split to"
        " apply the quantum filter. The hilbert space is better exploit "
        " if we set n^3=2^l.",
        "stride": "int. Stride used to move across the image.",
        "shots": "Number of shots per experiment (only Qiskit backends)",
        "backend": "Backend where to run the code. Either Qiskit or torch"
    }
    example_parameters = {
        "shape": (2, 2, 2),
        "stride": 2,
        "shots": 4096,
        "backend": "aer_simulator",
    }

    def __init__(
            self, shape: tuple = (4, 4, 4), stride: float = 1,
            shots: int = 4096, backend: str = 'torch'):

        super().__init__(
            n_dimensions=3, shape=shape, stride=stride,
            shots=shots, backend=backend)
