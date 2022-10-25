# Ingenii Quantum Hybrid Networks

Version: 0.1.0

Package of tools to integrate hybrid quantum-classical neural networks to your machine learning algorithms.
The algorithms provided in this package are implemented both in Qiskit (meant to run on real hardware and fake providers) and in Pytorch (meant to run in quantum simulation with CPUs or GPUs). This package contains the following quantum algorithms:

### Quantum convolutional layer (2D and 3D):
It is designed to reduce the complexity of the classical 2D/3D CNN, while maintaining its prediction performance. The hybrid CNN replaces a convolutional layer with a quantum convolutional layer. That is, each classical convolutional filter is replaced by a quantum circuit, which acts as a quantum filter. Each quantum circuit is divided into two blocks: the data encoding, which maps the input data into a quantum circuit, and the quantum transformation, where quantum operations are applied to retrieve information from the encoded data. Tha package contains an implementation for 2D data (images) and for 3D data (volumes).

### Quantum Hadamard Edge Detection (2D and 3D):
Performs edge detection for 2D data (images) and 3D data (volumes), using quantum operations. 
