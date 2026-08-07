# Biophyics-informed deep learning

A PyTorch-based framework for modeling continuous-time, continuous-depth dynamics in multi-area networks of laminar-resolved cortical columns using Ordinary Differential Equations (ODEs) and Stochastic Differential Equations (SDEs).

This project explores how biologically inspired neural networks can learn functional connectivity, lateral inhibition, and continuous dynamics across cortical columns composed of excitatory and inhibitory populations (`L23e`, `L23i`, `L4e`, `L4i`, `L5e`, `L5i`, `L6e`, `L6i`).

---

## Key Features

- **Continuous-Time Neural Dynamics**: Evaluates differential equations for membrane potentials and spike-frequency adaptation using numerical solvers from [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) (`odeint`, `odeint_adjoint`) and [`torchsde`](https://github.com/google-research/torchsde) (`sdeint`, `sdeint_adjoint`).
- **Modular Column Architecture**: Dynamically configure cortical areas (`BrainArea`), inter-area connections (`Connection`), receptive fields, and weight constraints (Dale's law).
- **Flexible Training**: Support for adjoint sensitivity methods for memory-efficient backpropagation through time.
- **Configurable Parameters**: Decoupled parameters managed cleanly via `.toml` files (`config/general_params.toml`, `config/model.toml`).

---

## Quick Start (Beginner Scripts)

If you are new to the codebase, start with these simple scripts:

### 1. [`scripts/example_network_setup.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/example_network_setup.py)
A hands-on walkthrough demonstrating how to:
- Instantiate a `BrainNetwork` from configuration files.
- Add cortical column areas (`add_area`).
- Connect areas with feedforward, feedback, lateral, input, and output connections.
- Run continuous simulations (`network.run`) and plot population firing rates (`network.analysis.plot_firing_rates`).

### 2. [`scripts/train_xor.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/train_xor.py)
A minimal training script demonstrating exclusive-OR (XOR) classification:
- Trains a 3-column network to solve the non-linearly separable XOR task.
- Optimizes connection weights using `torch.optim.Adam` and `odeint_adjoint`.

---

## Core Benchmark & Application Scripts

The main research tasks and training pipelines are organized in dedicated script subdirectories:

### 1. Winner-Take-All (WTA) Decision Making — [`scripts/wta/train_wta.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/wta/train_wta.py)
Trains a network of two cortical columns to exhibit winner-take-all perceptual decision-making dynamics driven by lateral inhibition. Firing rates in L2/3 excitatory populations are trained to match the canonical decision-making model of Wong & Wang (2006).

### 2. Context-Dependent Decision Making — [`scripts/context_wta/train_context_wta.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/context_wta/train_context_wta.py)
Extends winner-take-all dynamics to context-dependent decision-making. The network receives both sensory input and context signals that dynamically modulate column competition and selective action selection.

### 3. Parity Classification — [`scripts/parity/train_parity.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/parity/train_parity.py)
Trains a multi-column network to perform parity classification (odd vs. even) on binary input vectors. Evaluates feedforward and lateral inhibition weight learning under volatility and E/I balance loss penalties.

### 4. Handwritten Digits Classification — [`scripts/digits/train_digits.py`](file:///Users/administrator/Documents/Projects/ColumnModel/scripts/digits/train_digits.py)
Applies cortical column networks to image classification using the `scikit-learn` Digits dataset. Features 2D spatial grid receptive field connectivity, contrastive suppression penalties, and L2 regularization.

---

## Repository Structure

```
ColumnModel/
├── config/                        # TOML parameter configurations
│   ├── general_params.toml        # Global time constants, gain, threshold, noise
│   └── model.toml                 # Structural connection masks and initializations
├── src/                           # Core framework source code
│   ├── brain_network.py           # Main BrainNetwork container class
│   ├── dynamics/
│   │   └── network_dynamics.py    # ODE/SDE vector field evaluations (d/dt)
│   ├── simulation/
│   │   ├── network_simulator.py   # Simulation engine interfacing with torchdiffeq/torchsde
│   │   └── ode_wrapper.py        # PyTorch nn.Module wrapper for ODE integration
│   ├── structure/
│   │   ├── brain_area.py          # Cortical area module managing column populations
│   │   └── connection.py          # Synaptic connection module & Dale's law constraints
│   ├── analysis/
│   │   ├── network_analyzer.py    # Weight heatmaps, connection summaries, and plotting
│   │   └── network_readout.py     # Trajectory and classification readout extraction
│   ├── save_and_load/
│   │   └── network_archiver.py   # Checkpointing, architecture export/import
│   └── utils/                     # Loss functions, path helpers, seeds, interpolation
├── scripts/                       # Training and evaluation scripts
│   ├── example_network_setup.py   # Starter script: network construction & simulation
│   ├── train_xor.py               # Starter script: XOR classification
│   ├── wta/                       # Winner-take-all decision-making pipeline
│   ├── context_wta/               # Context-dependent decision-making pipeline
│   ├── parity/                    # Parity classification pipeline
│   └── digits/                    # Handwritten digits classification pipeline
├── data/                          # Target datasets and pre-computed ground truth trajectories
├── models/                        # Saved model checkpoints (.pt)
└── results/                       # Evaluation plots and training loss histories
```

---

## Installation & Requirements

Ensure you have Python 3.10+ installed along with the required dependencies:

```bash
pip install torch torchdiffeq torchsde numpy scipy matplotlib scikit-learn tomllib
```

---

## Basic Usage Example

```python
from src.brain_network import BrainNetwork

# 1. Load configuration and build network
network = BrainNetwork.from_toml("config/model.toml", "config/general_params.toml")

# 2. Add cortical areas and connections
network.add_area(area_name="v1", size=2)
network.add_area(area_name="v2", size=1)

network.add_input_connection(target_area="v1", input_size=2)
network.add_feedforward_connection(source="v1", target="v2")
network.add_output_connection(source_area="v2")

# 3. Run simulation
stimulus = [[15.0, 25.0]]
output = network.run(stimulus, adjoint=True, stochastic=False)

# 4. Extract readout & visualize firing rates
readout = network.read_out(output, mode="classification")
network.analysis.plot_firing_rates(output)
```

---

## Configuration

Model and general parameters are defined in the `config/` folder:
- **`config/general_params.toml`**: Defines global time constants ($\tau_m, \tau_s, \tau_a$), firing rate gain ($g$), threshold ($\theta$), noise factors ($\beta$), and background drive currents.
- **`config/model.toml`**: Specifies layer-specific population sizes, connection mask matrices, baseline synaptic strengths, and initial connectivity weights.

## To add
- Adding custom connections
- 
