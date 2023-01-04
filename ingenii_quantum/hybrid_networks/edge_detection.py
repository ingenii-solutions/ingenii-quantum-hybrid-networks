import numpy as np
import torch

from itertools import product, permutations
from math import sqrt
from qiskit import QuantumCircuit, Aer
from qiskit import transpile, assemble
from tqdm import tqdm
from time import time

from .utils import roll_numpy, roll_torch


class EdgeDetectorBase:
    def __init__(self, n_dimensions: int, size: int, backend: str, shots: int):

        self.n_dimensions = n_dimensions

        # 1. Circuit parameters: data qubits  ancilla qubits
        self.size = size
        self.shape = (size,) * n_dimensions
        self.data_qb = int(np.log2(self.size**n_dimensions))
        self.anc_qb = 1
        self.total_qb = self.data_qb + self.anc_qb
        self.backend = backend

        # Running in Pytorch
        if backend == 'torch':
            # 1. set CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(int("cuda:0".split(':')[1]))
            else:
                self.device = torch.device("cpu")

            # 2. Quantum operations: Hadamard and rolling Identity
            hadamard = 1/sqrt(2)*torch.tensor(
                [[1, 1], [1, -1]], dtype=torch.float32
            ).to(self.device)

            self.hadamard_large = torch.kron(
                torch.eye(2**self.data_qb), hadamard
            ).reshape(
                1, 2**self.total_qb, 2**self.total_qb
            ).to(self.device)

            # Initialize the amplitude permutation unitary
            self.D2n = torch.roll(
                torch.eye(2**self.total_qb, dtype=torch.float32), 1, 1
            ).reshape(
                1, 2**self.total_qb, 2**self.total_qb
            ).to(self.device)

        # Running in a Qiskit environment
        else:
            self.shots = shots
            # 2. Quantum operations: rolling Identity
            self.D2n = np.roll(np.identity(2**self.total_qb), 1, axis=1)

    def run_box(self, box, tol=1e-3):
        '''
        Run the edge detection algorithm for every box of the image
        (For Qiskit version)
            box (np.array): input data
            tol (np.array): Tolerance to be considered and edge
        '''
        # Create quantum circuit
        qc = QuantumCircuit(self.total_qb)

        # Data encoding: amplitude encoding
        box = box.flatten()
        initial_state = np.zeros(2**self.data_qb)
        initial_state[:box.shape[0]] = box
        initial_state = initial_state/np.linalg.norm(initial_state)
        qc.initialize(initial_state, range(1, self.total_qb))

        # Apply quantum operations
        qc.h(0)
        qc.unitary(self.D2n, range(self.total_qb))
        qc.h(0)

        # Measure
        qc.measure_all()

        # Run quantum circuit
        if self.backend == 'aer_simulator':
            aer_sim = Aer.get_backend(self.backend)
            t_qc = transpile(qc, aer_sim)
            qobj = assemble(t_qc, shots=self.shots)
            result = aer_sim.run(qobj).result()
        else:  # For real hardware/fake simulators
            # coupling_map = self.backend.configuration().coupling_map
            optimized_3 = transpile(
                qc, backend=self.backend, seed_transpiler=11)
            result = self.backend.run(optimized_3, shots=self.shots).result()
        counts = result.get_counts(qc)

        # Get statevector from counts
        # Calculate binary string of basis qubits
        # Let's calculate the ordered statevector
        new_counts = [
            counts.get(
                ''.join(map(str, qbits)),
                0
            )  # Generate string link e.g. '000', '001', ...
            for qbits in product([0, 1], repeat=self.total_qb)
        ]
        statevector = np.array(new_counts)/np.sum(new_counts)

        # Get odd values from state
        final_state = statevector[range(1, 2**(self.data_qb + 1), 2)]

        # Select values larger than threshold
        edge_scan = np.zeros(final_state.shape)
        edge_scan[np.abs(final_state) > tol] = 1

        # Revert to box shape
        result = edge_scan[:self.size**self.n_dimensions].reshape(self.shape)
        return result

    def run_image_qiskit(self, data, tol=1e-3, reduce=True, verbose=False):
        '''
        Run edge detection for the whole image (Qiskit version)
            data (np.array): input data
            tol (np.array): Tolerance to be considered an edge
            reduce (bool): Whether to reduce the dimension by half at the end
            of the algorithm
            verbose (bool): If true, tqdm is used to show the evolution.
        '''
        # 1. Get view of image
        # We split it into bits of (4x4x4)
        windows, shape_aux = roll_numpy(
            data, np.zeros((self.size,) * self.n_dimensions),
            dx=self.size, dy=self.size,
            dz=self.size if self.n_dimensions == 3 else None
        )
        windows = windows.reshape((-1,) + (self.size,) * self.n_dimensions)

        # 2. Run every box thorugh the QC
        def get_box_result(box):
            if np.sum(box) > tol:
                return self.run_box(box, tol)
            else:
                return np.zeros(self.shape)

        if verbose:
            results = [
                get_box_result(windows[i])
                for i in tqdm(range(windows.shape[0]))
            ]
        else:
            results = [
                get_box_result(windows[i])
                for i in range(windows.shape[0])
            ]

        # 3. Reshape image to original shape
        results = np.array(results).reshape(shape_aux)

        transpose_idxs_map = {
            4: [0, 2, 1, 3],
            5: [0, 1, 3, 2, 4],
            6: [0, 3, 1, 4, 2, 5],
            7: [0, 1, 4, 2, 5, 3, 6],
            8: [0, 1, 2, 5, 3, 6, 4, 7],
        }
        transpose_idxs = transpose_idxs_map[len(results.shape)]

        reverted_image = results.transpose(transpose_idxs).reshape(data.shape)

        if not reduce:
            return reverted_image

        # 4. Reduce image size
        size_x = data.shape[-1]
        reverted_image_small = reverted_image[
            :, :, :, :, range(1, size_x, 2)
        ][
            :, :, :, range(1, size_x, 2), :
        ]

        if self.n_dimensions == 3:
            reverted_image_small = \
                reverted_image_small[:, :, range(1, size_x, 2), :, :]

        return reverted_image_small

    def run_image_torch(self, data, tol=1e-3, reduce=True, verbose=False):
        '''
        Run edge detection for the whole images (Pytorch)
            data (tensor): input data
            tol (tensor): Tolerance to be considered and edge
            reduce (bool): reduce the dimension by half at the end of the
            algorithm
            verbose (bool): If true, tqdm is used to show the evolution.
        '''
        if verbose:
            start_time = time()

        if self.n_dimensions == 2:
            axis = (1, 2)
            reshape_idxs = (-1, 1, 1)
            normalized_reshape_idxs = (-1, self.size**self.n_dimensions)
            windows_reshape_idxs = (-1, self.size, self.size)
        else:
            axis = (2, 3, 4)
            samples = data.shape[0]
            reshape_idxs = (samples, -1, 1, 1, 1)
            normalized_reshape_idxs = \
                (samples, -1, self.size**self.n_dimensions)
            windows_reshape_idxs = \
                (samples, -1, self.size, self.size, self.size)

        # 1. Get view of image
        # We split it into bits of (4x4x4)
        windows, shape_aux = roll_torch(
            data, torch.zeros((self.size,) * self.n_dimensions),
            dx=self.size, dy=self.size,
            dz=self.size if self.n_dimensions == 3 else None
        )
        windows = windows.reshape(windows_reshape_idxs).to(self.device)

        # 2. Data encoding: amplitude encoding
        norm_const = windows.square().sum(axis=axis) \
            .sqrt().reshape(*reshape_idxs)
        norm_const[torch.abs(norm_const) < 1e-16] = 1
        normalized_image = windows/norm_const

        # 3. Flatten the last 2 dimesions to obtain an initial state
        normalized_flatten = normalized_image.reshape(normalized_reshape_idxs)

        # 4. Kronecker product to increase the last dimension 1 qubit
        normalized_larger = torch.kron(
            torch.tensor(normalized_flatten, dtype=torch.float32),
            torch.tensor([1, 0]).to(self.device)
        )

        # 5. Apply quantum operations
        final_state_h = (
            normalized_larger @ self.hadamard_large @ self.D2n
            @ self.hadamard_large
        )[:, :, range(1, 2**(self.data_qb+1), 2)]

        # 6. Select values larger than threshold
        edge_scan_h = torch.zeros(final_state_h.shape).to(self.device)
        edge_scan_h[torch.abs(final_state_h) > tol] = 1

        # 7. Reshape image to original shape
        permutations_idxs = {
            2: [0, 1, 3, 2, 4],
            3: [0, 1, 2, 5, 3, 6, 4, 7],
        }
        reverted_image = edge_scan_h \
            .reshape(shape_aux) \
            .permute(permutations_idxs[self.n_dimensions]) \
            .reshape(data.shape)
        if self.n_dimensions == 3:
            reverted_image[:, :, :, :, 0] = 0

        # 8. Reduce image size
        if reduce:
            size_x = data.shape[-1]
            reverted_image = reverted_image[
                :, :, :, :, range(1, size_x, 2)
            ][
                :, :, :, range(1, size_x, 2), :
            ]

            if self.n_dimensions == 3:
                reverted_image = \
                    reverted_image[:, :, range(1, size_x, 2), :, :]

        if verbose:
            print('Total execution time: ', time() - start_time)

        return reverted_image


###############################################################################
# QUANTUM EDGE DETECTION ALGORITHM FOR 2D IMAGES
###############################################################################

class EdgeDetector2D(EdgeDetectorBase):
    description = "Quantum Hadamard Edge Detection algorithm for 2D images. "
    "Implemented both with Qiskit to be run on real hardware, and on pytorch"
    " for quantum simulation"
    class_parameters = {
        "size": "(int) size of the (nxn) blocks in which the image is "
        "partitioned",
        "backend": "Qiskit backend (simulation or real) or torch",
        "shots": "(int) Number of shots for the experiments, only for "
        "qiskit backend",
        "anc_qb": " (int) Number of ancilla qubits",
        "total_qb": "(int) Total number of qubits",
        "device": "Pytorch device (cpu, cuda)",
        "H_large": "Id \tensordot H",
        "D2n": "Decrement gate",
    }
    required_parameters = {
        "size": "(int) size of the (nxn) blocks in which the image is "
        "partitioned",
    }
    optional_parameters = {
        "backend": "Qiskit backend (simulation or real) or torch",
        "shots": "(int) Number of shots for the experiments, only for qiskit "
        "backend",
    }
    example_parameters = {
        "size": 16,
        "backend": "torch",
        "shots": 512
    }

    def __init__(self, size, backend='aer_simulator', shots=1000):
        '''
        Quantum Hadamard Edge Detection class for 2D images,
            size (int): (int) size of the (nxn) blocks in which the image is
            partitioned
            backend: Qiskit backend (simulation or real) or torch
            shots (int): Number of shots for the experiments, only for qiskit
            backend
        '''

        super().__init__(
            n_dimensions=2, size=size, backend=backend, shots=shots)

    def run(self, data, tol=1e-3, reduce=True, verbose=False):
        """
        Runs the edge detection algorithm with either Pytorch or Qiskit code
            data (tensor/np.array): input data
            tol (tensor/np.array): Tolerance to be considered and edge
            reduce (bool): reduce the dimension by half at the end
            verbose (bool): If true, tqdm is used to show the evolution.
        """
        if self.backend == 'torch':
            return self.run_image_torch(data, tol, reduce, verbose)
        return self.run_image_qiskit(data, tol, reduce, verbose)


###############################################################################
# QUANTUM EDGE DETECTION ALGORITHM FOR 3D VOLUMES
###############################################################################

class EdgeDetector3D(EdgeDetectorBase):
    description = "Quantum Hadamard Edge Detection algorithm for 3D volumes. "
    "Implemented both with Qiskit to be run on real hardware, and on pytorch "
    "for quantum simulation"
    class_parameters = {
        "size": "(int) size of the (nxn) blocks in which the image is "
        "partitioned",
        "backend": "Qiskit backend (simulation or real) or torch",
        "shots": "(int) Number of shots for the experiments, "
        "only for qiskit backend",
        "anc_qb": " (int) Number of ancilla qubits",
        "total_qb": "(int) Total number of qubits",
        "device": "Pytorch device (cpu, cuda)",
        "H_large": "Id \tensordot H",
        "D2n": "Decrement gate",
    }
    required_parameters = {
        "size": "(int) size of the (nxn) blocks in which the image is "
        "partitioned",
    }
    optional_parameters = {
        "backend": "Qiskit backend (simulation or real) or torch",
        "shots": "(int) Number of shots for the experiments, "
        "only for qiskit backend",
    }
    example_parameters = {
        "size": 16,
        "backend": "torch",
        "shots": 512
    }

    def __init__(self, size, backend='aer_simulator', shots=100):
        '''
        Quantum Hadamard Edge dectection class for 3D images
            size (int): size of blocks
            backend: Qiskit backend, or torch
            shots (int): Number of experiments to run, only for Qiskit backend
        '''

        super().__init__(
            n_dimensions=3, size=size, backend=backend, shots=shots)

    def run(self, data, num_filters=6, tol=1e-3, reduce=True, verbose=False):
        '''
        Run edge detection for different rotations of the image.
            data (np.array): input data (samples, features, dim,dim,dim)
            num_filters (int): Number of rotations (permutations of the last 3
            dimension) to output
            tol (float):Tolerance to consider an edge
            reduce (bool): reduce the size of the image at the end
            verbose (bool): If true, tqdm is used to show the evolution.
        '''
        if reduce:
            fin_shape = int(data.shape[-1] / 2)
        else:
            fin_shape = int(data.shape[-1])
        perm = list(permutations([2, 3, 4]))
        if self.backend == 'torch':
            data_out = torch.zeros((
                data.shape[0], data.shape[1]*num_filters,
                fin_shape, fin_shape, fin_shape))
        else:
            data_out = np.zeros((
                data.shape[0], data.shape[1]*num_filters,
                fin_shape, fin_shape, fin_shape))
        for i in range(num_filters):
            p = [0, 1] + list(perm[i])
            if self.backend == 'torch':
                data_out[:, data.shape[1]*i:data.shape[1]*(i+1), :, :, :] = \
                    self.run_image_torch(
                        data.permute(p), tol, reduce, verbose).permute(p)
            else:
                data_out[:, data.shape[1]*i:data.shape[1]*(i+1), :, :, :] = \
                    self.run_image_qiskit(
                        data.transpose(p), tol, reduce, verbose).transpose(p)
        return data_out
