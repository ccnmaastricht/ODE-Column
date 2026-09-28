import torch
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.utils.paths import config_path, models_path
from src.utils.set_seed import set_seed
from train_memory import initialize_self_excitation_connection, initialize_lat_in_connection



def show_adaptation(network):

    output1 = network.run([0., 20.], input_window=[0.5, 1.0])
    output2 = network.run([0., 20.], input_window=[0.0, 1.0], reset_state=False)
    output3 = network.run([0., 0.], input_window=[0.0, 1.0], reset_state=False)
    output4 = network.run([20., 20.], input_window=[0.0, 0.75], reset_state=False)
    output = torch.concat([output1, output2, output3, output4], dim=0)

    network.analysis.plot_firing_rates(output, population='L23e')

def show_weak_perception(network):

    output1 = network.run([0., 0.], input_window=[0.0, 1.0])
    output2 = network.run([0., 10.], input_window=[0.0, 0.5], reset_state=False)
    output3 = network.run([0., 0.], input_window=[0.0, 0.5], reset_state=False)
    output4 = network.run([20., 20.], input_window=[0.0, 0.75], reset_state=False)
    output = torch.concat([output1, output2, output3, output4], dim=0)

    network.analysis.plot_firing_rates(output, population='L23e')

def show_adaptation_and_weak_perception(network):

    output1 = network.run([0., 20.], input_window=[0.5, 1.0])
    output2 = network.run([0., 20.], input_window=[0.0, 1.0], reset_state=False)
    output3 = network.run([0., 10.], input_window=[0.0, 0.5], reset_state=False)
    output4 = network.run([20., 20.], input_window=[0.0, 0.75], reset_state=False)
    output = torch.concat([output1, output2, output3, output4], dim=0)

    network.analysis.plot_firing_rates(output, population='L23e')

def show_longer_adaptation(network):

    output1 = network.run([0., 0.], input_window=[0.0, 3.0], stochastic=train_with_noise)
    output2 = network.run([0., 20.], input_window=[0.0, 3.0], reset_state=False, stochastic=train_with_noise)
    output3 = network.run([0., 0.], input_window=[0.0, 3.0], reset_state=False, stochastic=train_with_noise)
    output4 = network.run([20., 20.], input_window=[0.0, 0.75], reset_state=False, stochastic=train_with_noise)
    output = torch.concat([output1, output2, output3, output4], dim=0)

    network.analysis.plot_firing_rates(output, population='L23e')

def get_data_batch(batch_size, stim_strength=20.):
    """"""

    adapter_versions = torch.tensor([
        [0., stim_strength],
        [stim_strength, 0.]
    ])
    adapter_stims = torch.tile(adapter_versions, (batch_size//2, 1))
    adapter_stims = adapter_stims[torch.randperm(adapter_stims.size(0))]

    imagery_input = adapter_stims

    br_stims = torch.tensor([
        [stim_strength, stim_strength]
    ])
    br_stims = torch.tile(br_stims, (batch_size, 1))

    return adapter_stims, imagery_input, br_stims

def train_br_img_exp(
        seed,
        nr_iterations=100,
        batch_size=64,
        lr=1e-1,
        train_with_noise=True,
        train_with_adjoint=True):
    """"""
    set_seed(seed)

    # Initialize network
    config = config_path('br_img_params.toml')
    network = BrainNetwork.from_toml(config)

    network.add_area('v1', 2)
    network.add_input_connection('v1', 2, unique_id='bottom_up',
                                 receptive_field_size=1, stride=1, std=0.0, trainable=False)
    network.add_input_connection('v1', 2, unique_id='top_down',
                                 trainable=True)
    network.add_custom_connection(connection_name='lateral_inhibition', source='v1', target='v1',
                                  initializer=initialize_lat_in_connection, trainable=False)
    network.add_custom_connection(connection_name='self_excitation', source='v1', target='v1',
                                  initializer=initialize_self_excitation_connection, trainable=False)
    network.add_output_connection('v1')

    # show_longer_adaptation(network)
    # show_adaptation(network)
    # show_weak_perception(network)
    # show_adaptation_and_weak_perception(network)


    # Initialize optimizer and activation margin
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    margin = 0.1


    def run_br_batch(itr):
        """"""
        adapter_stims, imagery_input, br_stims = get_data_batch(batch_size)

        # Exp sequence: 3s rest -> 3s adapter -> 3s imagery -> 0.75s binocular rivalry -> 2.15s rest
        resting_state = network.run({'bottom_up': torch.zeros_like(adapter_stims),
                                     'top_down': torch.zeros_like(imagery_input)},
                                    stochastic=train_with_noise, adjoint=train_with_adjoint)
        adap_activity = network.run({'bottom_up': adapter_stims, 'top_down': torch.zeros_like(imagery_input)},
                                    input_window=[0.0, 3.0], reset_state=False,
                                    stochastic=train_with_noise, adjoint=train_with_adjoint)
        img_activity = network.run({'top_down': imagery_input, 'bottom_up': torch.zeros_like(adapter_stims)},
                                   input_window=[0.0, 3.0], reset_state=False,
                                   stochastic=train_with_noise, adjoint=train_with_adjoint)
        br_activity = network.run({'bottom_up': br_stims, 'top_down': torch.zeros_like(imagery_input)},
                                  input_window=[0.0, 0.75], reset_state=False,
                                  stochastic=train_with_noise, adjoint=train_with_adjoint)
        output = torch.concat([resting_state, adap_activity, img_activity, br_activity], dim=0)
        if itr > 1:
            network.analysis.plot_firing_rates(output, population='L23e')

        model_activations = network.read_out(br_activity, mode='classification')

        target_class = torch.argmax(adapter_stims, dim=1)

        batch_idx = torch.arange(model_activations.shape[0], device=model_activations.device)
        correct = model_activations[batch_idx, target_class]

        masked = model_activations.clone()
        masked[batch_idx, target_class] = -torch.inf
        incorrect = masked.max(dim=1).values

        loss = torch.relu(margin - correct + incorrect).mean()

        # print(model_activations)
        # print(target_class)
        # print(correct)
        # print(incorrect)
        # print(margin - correct + incorrect)
        # print(loss.item())

        return loss


    # Training loop
    for itr in range(nr_iterations):
        optimizer.zero_grad()

        loss = run_br_batch(itr)

        loss.backward()
        optimizer.step()

        print('Iter {:02d} | Train loss | {:.5f}'.format(itr, loss.item()))




if __name__ == '__main__':

    seed = 1
    train_with_noise = True
    train_with_adjoint = False

    train_br_img_exp(
        seed,
        train_with_noise=train_with_noise,
        train_with_adjoint=train_with_adjoint
    )


