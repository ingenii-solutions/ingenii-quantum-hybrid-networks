import random
import numpy as np
import torch
import torch.nn as nn
import math

import collections
import qiskit 
from qiskit import QuantumCircuit, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit import BasicAer
from qiskit import transpile, assemble
from qiskit.quantum_info import Statevector
import itertools
from qiskit.quantum_info import Pauli
from qiskit.opflow import *
from tqdm import tqdm
import time

#######################################################################################################################################################################
# QUANTUM EDGE DETECTION ALGORITHM FOR 2D IMAGES
#######################################################################################################################################################################


# Rolling 2D window for ND array
def roll2D(a,      
         b,      
         dx=1,   
         dy=1):   

    '''
        Rolling 2D window for ND array
            a (np.array): input array
            b (np.array): rolling 2D window array
            dx (int): horizontal step, abscissa, number of columns
            dy (int): vertical step, ordinate, number of rows
        '''
    shape = a.shape[:-2] + \
            ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
            ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
            b.shape  # multidimensional "sausage" with 2D cross-section
    strides = a.strides[:-2] + \
            (a.strides[-2] * dy,) + \
            (a.strides[-1] * dx,) + \
              a.strides[-2:]

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides), shape

# Rolling 2D window for ND array
def roll2DTorch(a,     
         b,      
         dx=1,  
         dy=1):  
    '''
        Rolling 2D window for torch
            a (np.array): input tensor
            b (np.array): rolling 2D window array
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

    return torch.as_strided(a, shape, strides), shape

class EdgeDetector2D():
    description = "Quantum Hadamard Edge Detection algorithm for 2D images. Implemented both with Qiskit to be run on real hardware, and on pytorch for quantum simulation"
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
        "size": "16",
        "backend": "torch",
        "shots": "512"
    }
    
    def __init__(self, size, backend='aer_simulator', shots=1000):  
        '''
        Quantum Hadamard Edge Detection class for 2D images,
            size (int): (int) size of the (nxn) blocks in which the image is partitioned
            backend: Qiskit backend (simulation or real) or torch
            shots (int): Number of shots for the experiments, only for qiskit backend
        '''
        # 1. Circuit parameters: data qubits  ancilla qubits
        self.size = size
        self.shape = (size,size)
        self.data_qb = int(np.log2(self.size**2))
        self.anc_qb = 1
        self.total_qb = self.data_qb + self.anc_qb
        self.backend = backend
        
        # Running in Pytorch
        if self.backend=='torch': 
            # set CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(int("cuda:0".split(':')[1]))
            else:
                self.device = torch.device("cpu")
            # 2. Quantum operations: Hadamard and rolling Identity
            H = 1/math.sqrt(2)*torch.tensor([[1,1],[1,-1]], dtype=torch.float32).to(self.device)
            self.H_large = torch.kron(torch.eye(2**self.data_qb), H).reshape(1,2**self.total_qb, 2**self.total_qb).to(self.device)
            # Initialize the amplitude permutation unitary
            self.D2n = torch.roll(torch.eye(2**self.total_qb, dtype=torch.float32), 1, 1).reshape(1,2**self.total_qb,2**self.total_qb).to(self.device)
        
        # Running in a Qiskit environment
        else:  
            self.shots=shots
            # 2. Quantum operations: rolling Identity
            self.D2n = np.roll(np.identity(2**self.total_qb), 1, axis=1)


    def run_box(self, box, tol=1e-3):
        '''
        Run the edge detection algorithm for every box of the image (For Qiskit version)
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
        if self.backend=='aer_simulator':
            back = Aer.get_backend(self.backend)
            t_qc = transpile(qc, back)
            qobj = assemble(t_qc, shots=self.shots)
            result = back.run(qobj).result()
        else:
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
        final_state = statevector[range(1,2**(self.data_qb+1),2)]
        
        # Select values larger than threshold
        edge_scan = np.zeros(final_state.shape)
        edge_scan[np.abs(final_state)>tol]=1
        
        # Revert to box shape
        result = edge_scan[:self.size**2].reshape(self.shape)
        return result
        
        
    def run_imageQiskit(self, data, tol=1e-3, reduce=True):
        '''
        Run edge detection for the whole image (Qiskit version)
            box (np.array): input data
            tol (np.array): Tolerance to be considered and edge
            reduce (bool): reduce the dimension by half at the end of the algorithm
        '''
        # 1. Get view of image
        # We split it into bits of (4x4x4)
        samples = data.shape[0]
        windows, shape_aux = roll2D(data,torch.zeros((self.size,self.size)),dx=self.size,dy=self.size)
        windows = windows.reshape(-1,self.size,self.size)
       
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
        if len(results.shape)==4:
            reverted_image = results.transpose([0, 2, 1, 3]).reshape(data.shape)
        if len(results.shape)==5:
            reverted_image = results.transpose([0, 1, 3, 2, 4]).reshape(data.shape)
        if len(results.shape)==6:
            reverted_image = results.transpose([0, 3, 1, 4, 2, 5]).reshape(data.shape)
            
        if not reduce:
            return reverted_image
        # 4. Reduce image size
        size_x = data.shape[-1]
        reverted_image_small = reverted_image[:,:,:,:,range(1,size_x,2)][:,:,:,range(1,size_x,2),:]
        return reverted_image_small
            
        return data_out
    
    def run_imageTorch(self, data, tol=1e-3, reduce=True):
        '''
        Run edge detection for the whole images (Pytorch version=
            data (tensor): input data
            tol (tensor): Tolerance to be considered and edge
            reduce (bool): reduce the dimension by half at the end of the algorithm
        '''
        if self.verbose:
            start_time = time.time()
        # 1. Get view of image
        # We split it into bits of (4x4x4)
        windows, shape_aux = roll2DTorch(data,torch.zeros((self.size,self.size)),dx=self.size,dy=self.size)
        windows = windows.reshape(-1,self.size,self.size).to(self.device)
        
        # 2. Data encoding: amplitude encoding
        norm_const = windows.square().sum(axis=(1,2)).sqrt().reshape(-1,1,1)
        norm_const[torch.abs(norm_const)<1e-16] = 1
        normalized_image = windows/norm_const
        
        # 3. Flatten the last 2 dimesions to obtain an initial state
        normalized_flatten = normalized_image.reshape(-1,self.size*self.size)
        # 4. Kronecker product to increase the last dimension 1 qubit
        
        normalized_larger = torch.kron(torch.tensor(normalized_flatten, dtype=torch.float32),torch.tensor([1,0]).to(self.device))
        # 7. Apply quantum operations
        final_state_h = (normalized_larger @ self.H_large @ self.D2n @ self.H_large)
        final_state_h =final_state_h[:,:,range(1,2**(self.data_qb+1),2)]
        # 8. Select values larger than threshold
        edge_scan_h = torch.zeros(final_state_h.shape).to(self.device)
        edge_scan_h[torch.abs(final_state_h)>tol]=1
        
        # 9. Reshape image to original shape
        reverted_image = edge_scan_h.reshape(shape_aux).permute([0, 1, 3, 2, 4]).reshape(data.shape)
        
        # 10. Reduce image size
        if reduce:
            size_x = data.shape[-1]
            reverted_image_small = reverted_image[:,:,:,:,range(1,size_x,2)][:,:,:,range(1,size_x,2),:]
            return reverted_image_small
        if self.verbose:
            end_time = time.time() - start_time
            print('Total execution time: ', end_time)
        return reverted_image
    
    def run(self, data, tol=1e-3, reduce=True, verbose=0):
        """
            Runs the edge detection algorithm with either Pytorch code or Qiskit code
            data (tensor/np.array): input data
            tol (tensor/np.array): Tolerance to be considered and edge
            reduce (bool): reduce the dimension by half at the end of the algorithm
            verbose (int): If 1, tqdm is used to show the evolution.
        """
        self.verbose = verbose
        if self.backend=='torch':
            return self.run_imageTorch(data, tol, reduce)
        return self.run_imageQiskit(data, tol, reduce)