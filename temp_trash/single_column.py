from temp_trash.column_network_wta import ColumnAreaWTA
from src.utils import *
from torchdiffeq import odeint



# --- Initialize network --- #
num_columns = 1
num_populations_first_area = 8

col_params = load_config('../config/single_params.toml')

network = ColumnAreaWTA(col_params, area='v1', num_columns=num_columns)

stim_duration = 0.5
dt = 1e-3
time_steps = int(stim_duration * 2 / dt)
time_vec = torch.linspace(0., time_steps * dt, time_steps)

initial_state = torch.zeros(1, num_columns * 8 * 2)  # 2 state variables
membrane_init = torch.tensor([-1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01, -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01])
initial_state[:, :num_populations_first_area] = torch.tile(membrane_init, (1, num_populations_first_area//8,))
network.set_time_vec(time_vec)


# --- Set stimulus --- #
stim = torch.tensor([[0., 0., 20., 20., 0., 0., 0., 0.]])
network.set_stim(stim)

# --- Run network and plot --- #
ode_output = odeint(network, initial_state, time_vec)

split = num_populations_first_area
firing_rates = compute_firing_rate(ode_output[:, :, :split] - ode_output[:, :, split:(split * 2)])
fr = firing_rates.detach().numpy()

layers = ['L2/3e', 'L2/3i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

for i in range(fr.shape[-1]):
    plt.plot(fr[:, :, i], label=layers[i])
plt.legend()
plt.show()

