# Neural ODEs to learn connectivity between cortical columns

Exploration of neural ODEs and their value in modeling laminar-resolved cortical columns' activity and connectivity. 
Neural ODEs are continuous-depth models and use continuous-time dynamics instead of discrete layers. 

The script ``two_column_ode.py`` uses an ODE to learn the synaptic strength of the lateral connections between two laminar-resolved cortical columns, recreating winner-take-all dynamics by means of lateral inhibition. 
The ODE takes as a function the differential equations for updating the membrane potential and adaptation, from which the firing rate can be computed. 
The model can be trained either on the membrane potential or on the firing rates. 

Another use case of ODEs can be found in ``ode_bifurcation.py`` in which an ODE is trained to learn a spiral trajectory determined by supercritical Hopf bifurcation. 
Based on input mu, the trajectory can either spiral inwards or outwards. 
