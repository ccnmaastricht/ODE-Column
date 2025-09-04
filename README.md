# Neural ODEs to learn functional connectivity between cortical columns

Exploration of neural ordinary differential equations and their value in modeling the activity of networks build out of laminar-resolved cortical columns. 
Neural ODEs are continuous-depth models and use continuous-time dynamics instead of discrete layers. 

### Supercritical Hopf bifurcation (no cortical columns)
A simpler use case of neural ODEs can be found in ``scripts/ode_bifurcation.py`` in which a neural ODE is trained to learn a spiral trajectory determined by supercritical Hopf bifurcation. 
Based on input mu, the trajectory can either spiral inwards or outwards. 

### Winner-take-all decision-making by means of lateral inhibition
In ``scripts/wta_ode.py`` a network of two cortical columns is trained to exhibit winner-take-all dynamics typical in perceptual decision-making. 
The L2/3 excitatory firing rates are trained to match the decision-making model of Wong and Wang (2006). 
Lateral inhibition weights and self-excitation weights in L2/3e are trainable. 

### Exclusive-or classification
In ``scripts/xor_ode.py`` a network of three cortical columns is trained to perform exclusive-or classification. 
Two first columns A, B receive the binary input and have feedforward connectivity to a final column C. 
The L2/3 excitatory firing rates of column C are trained to make a binary decision and correctly classify the input. 
The feedforward weights between the input and columns A, B and between A, B and column C are trainable. 

### Odd/even classification
Finally, in ``scripts/parity.py`` a network is trained to perform parity classification (odd/even) on based on four input units. 
The final column's firing rates should make a binary decision (20Hz=input is even, 0Hz=input is odd). 
What is trainable are the feedforward connections between column-areas and the lateral inhibition connections within column-areas (lateral between columns). 

### Stochastic neural ODEs
All three column networks can both be trained with ``torchdiffeq``'s ``odeint`` function (deterministic) or with ``torchsde``'s ``sdeint`` function (stochastic; adds noise to the network activity). 
For the latter option, it should be noted that artefacts in firing rates can occur when running larger networks. 
These artefacts are avoided when the param ``adaptive`` is set to ``True``, but this significantly increases computation time. 


Model parameters can be changed in ``config/model.toml``. 
