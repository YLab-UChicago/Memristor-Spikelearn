# Memristor-Spikelearn

Memristor-Spikelearn is a simulator for memristor-based implementation of SNN synaptic plasticity.

More details about the design of this simulator can be found in our paper:  
Y. Liu, A. Yanguas-Gil, S. Madireddy and Y. Li, "Memristor-Spikelearn: A Spiking Neural Network Simulator for Studying Synaptic Plasticity under Realistic Device and Circuit Behaviors," 2023 Design, Automation & Test in Europe Conference & Exhibition (DATE)

## File structure

```
Memristor-Spikelearn
├── examples
│   ├── approximate_bp.py: MNIST classifier with approximate gradient descent
│   └── stdp_mnist.py: MNIST classifier based on STDP
├── mnist.npz
├── model_data
│   └── 1t1r_Paiyu_15.npz: look-up table specifying 1T1R synapse behavior
└── src
    └── spikelearn
          ├── __init__.py
          ├── generators.py
          ├── loihi.py
          ├── neurons.py: includes additional neurons needed to implement the examples
          ├── rules.py
          ├── snn.py
          ├── synapses.py: includes various models of memristor device and synapse circuits
          ├── trace.py
          └── transforms.py
```

## Demos

`examples/stdp_mnist.py`: This file implements a MNIST classifier based on spike timing dependent plasticity (STDP).

Usage:  
```
examples/stdp_mnist.py <# hidden neurons> <# samples in supervised labeling> <# epochs> Vteam <learning rate> <conductance corresponding to weight 1>
examples/stdp_mnist.py <# hidden neurons> <# samples in supervised labeling> <# epochs> Paiyu_chen_15 <maximum programming voltage> <conductance corresponding to weight 1> <model evaluation time steps per SNN simulation step>
examples/stdp_mnist.py <# hidden neurons> <# samples in supervised labeling> <# epochs> data-driven model_data/1t1r_Paiyu_15.npz <conductance corresponding to weight 1> <maximum potentiation voltage (on transistor gate)> <maximum depression voltage> 0.0105 0.0333
```

`examples/approximate_bp.py`: This file implements a MNIST classifier that uses synaptic plasticity to approximate gradient descent.

Usage:
```
examples/approximate_bp.py Vteam <learning rate> <conductance corresponding to weight 1>
examples/approximate_bp.py Paiyu_chen_15 <maximum programming voltage> <conductance corresponding to weight 1> <model evaluation time steps per SNN simulation step>
```
