# Ingenii Quantum Hybrid Networks

Version: 0.2.0

Package of tools to integrate hybrid quantum-classical neural networks to your machine learning algorithms.
The algorithms provided in this package are implemented both in Qiskit (meant to run on real hardware and fake providers) and in Pytorch or Tensorflow (meant to run in quantum simulation with CPUs or GPUs). This package contains the following quantum algorithms:

## Quantum convolutional layer (2D and 3D):
It is designed to reduce the complexity of the classical 2D/3D CNN, while maintaining its prediction performance. The hybrid CNN replaces a convolutional layer with a quantum convolutional layer. That is, each classical convolutional filter is replaced by a quantum circuit, which acts as a quantum filter. Each quantum circuit is divided into two blocks: the data encoding, which maps the input data into a quantum circuit, and the quantum transformation, where quantum operations are applied to retrieve information from the encoded data. Tha package contains an implementation for 2D data (images) and for 3D data (volumes).

## Quantum fully-connected layer
It is designed to construct hybrid quantum-classical neural networks, where the quantum layer is a fully-connected layer. The quantum layer is a parametrized quantum circuit, composed of three parts: a data encoding which maps the classical data into a quantum circuit, a parametrized quantum circuit, which performs quantum operations with parameters, and measurements, which produce the output of the layer. Multiple quantum architectures are provided, which have been extracted from previous studies of hybrid neural networks. **Update:** To improve efficiency of the training, the codes have been rewritten using Pennylane instead of Qiskit. You can still run the algorithm in Qiskit devices by providing the suitable backend name.  

## Quantum fusion model
It is designed to efficiently integrate the extracted features from two classical neural network models to produce enhanced predictions. The proposed model strategically integrates 3D-CNNs and SG-CNNs to leverage their respective strengths in processing diverse facets of the training data. 
The simulation results presented here will demonstrate the superior performance of the quantum fusion model relative to state-of-the-art classical models.

## Quantum Hadamard Edge Detection (2D and 3D):
Performs edge detection for 2D data (images) and 3D data (volumes), using quantum operations. 
