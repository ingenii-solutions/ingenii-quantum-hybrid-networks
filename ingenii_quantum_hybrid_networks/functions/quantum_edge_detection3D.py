import numpy as np
import torch
import math

from qiskit import QuantumCircuit, Aer
from qiskit import transpile, assemble
import itertools
from qiskit.opflow import *
from tqdm import tqdm
import time

from utils import roll_numpy, roll_torch

#######################################################################################################################################################################
# QUANTUM EDGE DETECTION ALGORITHM FOR 3D VOLUMES
#######################################################################################################################################################################

class EdgeDetector3D():
    description = "Quantum Hadamard Edge Detection algorithm for 3D volumes. Implemented both with Qiskit to be run on real hardware, and on pytorch for quantum simulation"
    class_parameters = {
         "size": "(int) size of the (nxn) blocks in which the image is partitioned",
          "backend": "Qiskit backend (simulation or real) or torch",
          "shots": "(int) Number of shots for the experiments, only for qiskit backend",
          "anc_qb": " (int) Number of ancilla qubits",
          "total_qb": "(int) Total number of qubits",
          "device": "Pytorch device (cpu, cuda)",
          "H_large": "Id \tensordot H",
          "D2n": "Decrement gate",
            }
    required_parameters = {
        "size": "(int) size of the (nxn) blocks in which the image is partitioned",
    }
    optional_parameters = {
        "backend": "Qiskit backend (simulation or real) or torch",
        "shots": "(int) Number of shots for the experiments, only for qiskit backend",
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
        # 1. Circuit parameters: data qubits  ancilla qubits
        self.size = size
        self.shape = (size,size,size)
        self.data_qb = int(np.log2(self.size**3))
        self.anc_qb = 1
        self.total_qb = self.data_qb + self.anc_qb
        self.backend = backend

        # Pytorch execution
        if backend=='torch':
             # set CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(int("cuda:0".split(':')[1]))
            else:
                self.device = torch.device("cpu")
            # 2. Quantum operations: Hadamard and rolling Identity
            H = 1/math.sqrt(2)*torch.tensor([[1,1],[1,-1]], dtype=torch.float32).to(self.device)
            self.H_large = torch.kron(
                torch.eye(2**self.data_qb), H
            ).reshape(
                1, 2**self.total_qb, 2**self.total_qb
            ).to(self.device)
            
            # Initialize the amplitude permutation unitary
            self.D2n = torch.roll(
                torch.eye(2**self.total_qb, dtype=torch.float32), 1, 1
            ).reshape(
                1, 2**self.total_qb, 2**self.total_qb
            ).to(self.device)
        
        # Qiskit execution
        else:
            self.shots=shots
            # 2. Quantum operations: rolling Identity
            self.D2n = np.roll(np.identity(2**self.total_qb), 1, axis=1)
                
            
    def run_box(self, box, tol=1e-3):
        '''
        Run quantum edge detection in a portion of the image
            box (np.array): input data (size,size,size)
            tol (float):Tolerance to consider an edge
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
        
        # Get statevector from counts
        # Calculate binary string of basis qubits
        qbit_list = list(itertools.product([0, 1], repeat=self.total_qb))
        new_counts = []
        for l in range(len(qbit_list)): # Let's calculate the ordered statevector
            q_str =''.join([str(value) for value in qbit_list[l]]) # Generate sting link '000', '001', ...
            new_counts.append(counts.get(q_str, 0)) # Get the counts from that string

        statevector = np.array(new_counts)/np.sum(new_counts)
        # Gett odd values from state
        final_state = statevector[range(1, 2**(self.data_qb+1), 2)]
        
        # Select values larger than threshold
        edge_scan = np.zeros(final_state.shape)
        edge_scan[np.abs(final_state)>tol]=1
        
        # Revert to box shape
        result = edge_scan[:self.size**3].reshape(self.shape)
        return result
        
        
    def run_image(self, data, tol=1e-3, reduce=True):
        '''
        Run quantum edge detection in the wholw image orset of images
            data (np.array): input data (samples, features, dim, dim, dim)
            tol (float):Tolerance to consider an edge
            reduce (bool): reduce the size of the image at the end
        '''
        # 1. Get view of image
        # We split it into bits of (4x4x4)
        samples = data.shape[0]
        windows, shape_aux = roll_numpy(
            data, np.zeros((self.size,self.size,self.size)),
            dx=self.size, dy=self.size, dz=self.size
        )
        windows = windows.reshape(-1,self.size,self.size,self.size)
        
        # 2. Run every box thorugh the QC
        results = []
        if self.verbose==1:
            for i in tqdm(range(windows.shape[0])):
                box = windows[i]
                if np.sum(box)>tol:
                    result = self.run_box(box, tol)
                else:
                    result = np.zeros(self.shape)
                results.append(result)
        else:
            for i in range(windows.shape[0]):
                box = windows[i]
                if np.sum(box)>tol:
                    result = self.run_box(box, tol)
                else:
                    result = np.zeros(self.shape)
                results.append(result)
            
        # 3. Reshape image to original shape
        results = np.array(results).reshape(shape_aux)
        if len(results.shape)==6:
            reverted_image = results.transpose([0, 3, 1, 4, 2, 5]).reshape(data.shape)
        elif len(results.shape)==7:
            reverted_image = results.transpose([0, 1, 4, 2, 5, 3, 6]).reshape(data.shape)                
        elif len(results.shape)==8:
            reverted_image = results.transpose([0, 1, 2, 5, 3, 6, 4, 7]).reshape(data.shape)         
        # 4. Reduce image size
        if reduce:
            size_x = data.shape[-1]
            reverted_image_small = reverted_image[:,:,:,:,range(1,size_x,2)][:,:,:,range(1,size_x,2),:][:,:,range(1,size_x,2),:,:]
            return reverted_image_small
        return reverted_image
    
    def run_imageTorch(self, data, tol=1e-3, reduce=True):
        '''
            Run edge detection for a set of images
                data (tensor): input data (samples, features, dim,dim,dim)
                tol (float): Tolerance to consider an edge
                reduce (bool): reduce the size of the image at the end
        '''
        if self.verbose:
            start_time = time.time()
        # 1. Get view of image
        # We split it into bits of (4x4x4)
        samples = data.shape[0]
        rolling_image, shape_aux = roll_torch(
            data, torch.zeros((self.size,self.size,self.size)),
            dx=self.size, dy=self.size, dz=self.size
        )
        rolling_image = rolling_image.reshape(samples,-1,self.size,self.size,self.size).to(self.device)
        
        # 2. Data encoding: amplitude encoding
        norm_const = rolling_image.square().sum(axis=(2,3,4)).sqrt().reshape(samples,-1,1,1,1)
        norm_const[norm_const<1e-16] = 1
        normalized_image = rolling_image/norm_const
        
        # 3. Flatten the last 2 dimesions to obtain an initial state
        normalized_flatten = normalized_image.reshape(samples,-1,self.size*self.size*self.size)
        # 4. Kronecker product to increase the last dimension 1 qubit
        normalized_larger = torch.kron(torch.tensor(normalized_flatten),torch.tensor([1,0]).to(self.device))
        
        # 7. Apply quantum operations
        final_state_h = (normalized_larger @ self.H_large @ self.D2n @ self.H_large)[:,:,range(1,2**(self.data_qb+1),2)]
        
        # 8. Select values larger than threshold
        edge_scan_h = torch.zeros(final_state_h.shape).to(self.device)
        edge_scan_h[torch.abs(final_state_h)>tol]=1
        
        # 9. Reshape image to original shape
        reverted_image = edge_scan_h.reshape(shape_aux).permute([0, 1, 2, 5, 3, 6, 4, 7]).reshape(data.shape)
        reverted_image[:,:, :,:,0]=0
        
        # 10. Reduce image size
        if reduce:
            size_x = data.shape[-1]
            reverted_image_small = reverted_image[:,:,:,:,range(1,size_x,2)][:,:,:,range(1,size_x,2),:][:,:,range(1,size_x,2),:,:]
            return reverted_image_small
        if self.verbose:
            end_time = time.time() - start_time
            print('Total execution time: ', end_time)
        return reverted_image
    
    def run(self, data, num_filters =6, tol=1e-3, reduce=True, verbose=0):
        '''
        Run edge detection for different rotations of the image.
            data (np.array): input data (samples, features, dim,dim,dim)
            num_filters (int): Number of rotations (permutations of the last 3 dimension) to output
            tol (float):Tolerance to consider an edge
            reduce (bool): reduce the size of the image at the end
            verbose (int): If 1, tqdm is used to show the evolution.
        '''
        self.verbose = verbose
        if reduce:
            fin_shape = int(data.shape[-1]/2)
        else:
            fin_shape = int(data.shape[-1])
        perm = list(itertools.permutations([2,3,4]))
        if self.backend=='torch':
            data_out = torch.zeros((data.shape[0], data.shape[1]*num_filters,fin_shape,fin_shape,fin_shape))
        else:
            data_out = np.zeros((data.shape[0], data.shape[1]*num_filters,fin_shape,fin_shape,fin_shape))
        for i in range(num_filters):
            p = [0,1]+list(perm[i])
            if self.backend=='torch':
                data_out[:,data.shape[1]*i:data.shape[1]*(i+1),:,:,:] = self.run_imageTorch(data.permute(p), tol, reduce).permute(p)
            else:
                data_out[:,data.shape[1]*i:data.shape[1]*(i+1),:,:,:] = self.run_image(data.transpose(p), tol, reduce).transpose(p)        
        return data_out