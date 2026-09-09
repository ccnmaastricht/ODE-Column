# Biophysics-informed deep learning

This repository provides the modeling framework to construct, train and evaluate 
biophysics-informed deep neural networks, effectively bridging bottom-up and top-down modeling approaches.
These networks consist of neuroanatomically realistic cortical columns (/canonical 
microcircuits) and can be trained by learning the functional connectivity profile between 
columns to exhibit meaningful, goal-directed behaviour.

---

## Key Features

### Biologically Grounded Architecture
- **Laminar cortical columns as network nodes**: Each node in the network is a canonical
  microcircuit composed of four cortical layers (L2/3, L4, L5, L6), each with a separate
  excitatory and inhibitory population, with a connectivity structure derived from
  neuroanatomical datasets ([Potjans & Diesmann, 2014](https://doi.org/10.1093/cercor/bhs358)).
- **Dale's law is respected**: A single neuronal population cannot both excite and inhibit its targets.
- **Continuous, time-resolved dynamics**: Network state is integrated continuously in time using
  adaptive numerical solvers from [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq)
  and [`torchsde`](https://github.com/google-research/torchsde), rather than discrete
  timesteps.
- **Realistic interacting state variables**: Each population tracks three coupled state
  variables (membrane potential, rate adaptation and firing rate)
  producing rich and realistic dynamics. 

### Training Flexibility & Engineering
- **Train on any cognitive task**: Connection weights are optimized with standard PyTorch
  optimizers and backpropagation through the ODE solver, so the network can in principle
  be trained to perform any task for which a differentiable loss can be defined.
- **Modular network construction**: Areas and connections can be added and configured
  in a straightforward manner and changing the architecture requires no modifications to the dynamics
  or training code.
- **Control over what is learned**: Each connection type (feedforward, feedback,
  lateral, input, output) can independently be set as trainable or fixed, giving full
  control over which parts of the network can be adjusted during learning.
- **Memory-efficient training via adjoint sensitivity**: `odeint_adjoint` and
  `sdeint_adjoint` recompute the forward trajectory during the backward pass rather than
  storing it, keeping computational overhead low.
- **Optional stochastic dynamics**: Switching to SDE mode adds Gaussian noise to the
  state dynamics.
- **Saving and loading**: Trained networks can be saved and restored via built-in
  checkpointing.

---

## Quick Start

### Basic Usage Example

```python
from src.brain_network import BrainNetwork

# 1. Load configuration and initialize network
network = BrainNetwork.from_toml(
  "config/example_params.toml",  # application-specific params (see Setting Parameters for more info)
  "config/general_params.toml"   # general model params (see Setting Parameters for more info)
)

# 2. Add brain areas and connections
network.add_area(area_name="v1", size=2)
network.add_area(area_name="v2", size=1)

network.add_input_connection(target_area="v1", input_size=2)
network.add_feedforward_connection(source="v1", target="v2")
network.add_output_connection(source_area="v2")

# Optional: Check initialized network connections
network.analysis.summarize_connections()
network.analysis.visualize_weights()

# 3. Run simulation
stimulus = [[15.0, 25.0]]  # stim size = (batch_size, input_size)
output = network.run(stimulus, adjoint=False, stochastic=True)

# 4. Examine network output
# 4a. Obtain model predictions
readout = network.read_out(output, mode="classification")  
# 4b. Transform raw output state (= membrane potential, adaptation) to firing rates
firing_rates = network.get_firing_rates(output)
# 4c. Plot firing rates
network.analysis.plot_firing_rates(output)
```

### Training Example: XOR Classification

A minimal training script demonstrating exclusive-OR (XOR) classification: [`scripts/train_xor.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/train_xor.py).

### Setting Parameters in `config/`

Initializing a `BrainNetwork` instance requires two configuration files located in the `config/` folder 
containing network parameters: 
- **Model-specific (application-specific) parameters**
  - Should contain at least `[time_params]dt` and `[time_params]sim_time` to allow network simulation and should 
    contain the appropriate connection weight initializations and masks (e.g. feedforward, lateral) for network 
    construction. 
- **General parameters**
  - Contains parameters that are generally applicable to all network applications, e.g. time constants, population 
    sizes, microcircuit connection probabilities. 
  - In case the user provides no configuration file, `BrainNetwork` initialization uses `general_params.toml` as a 
    default. 

### Investigating Network Weights

When a connection is added (e.g. with `add_feedforward_connection()`), its weights are
automatically initialized and registered as a `torch.nn.Parameter`. Each `Connection`
instance holds two weight matrices:
- `Connection.weights`: the raw, unconstrained parameter tracked by PyTorch autograd.
  Use this when registering weights with an optimizer.
- `Connection.W`: the effective weight matrix used in the network dynamics. It is
  sign-constrained (to enforce Dale's law) and masked to the connection's allowed connections.
  **Use this when inspecting or visualizing learned weights.**


Weights can be accessed in three ways:
```
BrainNetwork.analysis.get_weights("feedforward_v1_v2")           # constrained W, by name
BrainNetwork.connections["feedforward_v1_v2"].W                  # constrained W, direct
BrainNetwork.connections["feedforward_v1_v2"].weights            # raw torch.nn.Parameter
```

For an overview of all network connections and their properties (name, source, target, size, trainable, etc.), run 
`BrainNetwork.analysis.summarize_connections()` after adding all connections. 


---

## Application Scripts

The main modeling applications are organized in dedicated script subdirectories:

### 1. Winner-Take-All (WTA) Decision-Making
[`scripts/wta/train_wta.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/wta/train_wta.py) 

Trains a network of two cortical columns to exhibit winner-take-all perceptual decision-making dynamics driven by lateral inhibition. 

### 2. Context-Dependent Decision Making 
[`scripts/context_wta/train_context_wta.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/context_wta/train_context_wta.py)

Extends winner-take-all dynamics to context-dependent decision-making. The four-column network receives both sensory input and context signals that together inform column competition.

### 3. Parity Classification 
[`scripts/parity/train_parity.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/parity/train_parity.py)

Trains a multi-area columnar network to perform parity classification (odd vs. even) on binary input vectors. 

### 4. Handwritten Digit Classification 
[`scripts/digits/train_digits.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/digits/train_digits.py)

Trains a multi-area columnar network with receptive-field connectivity to perform image classification using the `scikit-learn` Digits dataset. 

---

## Repository Structure

```
ColumnModel/
├── config/                        # TOML parameter configurations
├── src/                           # Core framework source code
│   ├── brain_network.py           # Main BrainNetwork container class
│   ├── dynamics/
│   │   └── network_dynamics.py    # Network state dynamics (ODE/SDE vector field evaluations)
│   ├── simulation/
│   │   ├── network_simulator.py   # Simulation engine interfacing with torchdiffeq/torchsde
│   │   └── ode_wrapper.py         # PyTorch nn.Module wrapper for ODE integration
│   ├── structure/
│   │   ├── brain_area.py          # Cortical area module
│   │   └── connection.py          # Synaptic connection module; stores network weights
│   ├── analysis/
│   │   ├── network_analyzer.py    # Weight visualization, connection summaries, and plotting
│   │   └── network_readout.py     # Trajectory and classification readout extraction
│   ├── save_and_load/
│   │   └── network_archiver.py    # Saving and restoring network checkpoints
│   └── utils/                     # Loss functions, path helpers, etc.
├── scripts/                       # Training and evaluation scripts
│   ├── basic_usage_example.py     # Starter script: network setup and usage
│   ├── train_xor.py               # Starter script: XOR classification
│   ├── wta/                       # Winner-take-all decision-making pipeline
│   ├── context_wta/               # Context-dependent decision-making pipeline
│   ├── parity/                    # Parity classification pipeline
│   └── digits/                    # Handwritten digit classification pipeline
├── data/                          # Target datasets
├── models/                        # Saved model checkpoints (.pt) and training histories (.pt)
└── results/                       # Results figures
```

---

## Requirements

Ensure you have Python 3.10+ installed along with the required dependencies:

```bash
pip install torch torchdiffeq torchsde numpy scipy matplotlib scikit-learn tomllib
```
