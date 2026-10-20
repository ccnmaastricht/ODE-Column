import torch
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.utils.paths import models_path, config_path
from train_br_img import get_data_batch




def compute_congruency(adapter_stims, model_activations):

    adapter_stims_argmax = torch.argmax(adapter_stims, dim=1)
    br_responses = torch.argmax(model_activations, dim=1)

    return (adapter_stims_argmax == br_responses).sum().item() / len(adapter_stims)

def run_passive_br(network, nr_trials):

    _, _, br_stims = get_data_batch(batch_size=nr_trials, stim_strength=20.)

    output1 = network.run({'bottom_up': torch.zeros_like(br_stims)},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level)
    br_activity = network.run({'bottom_up': br_stims},
                          sim_time=1.0, input_window=[0.0, 0.75], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    # output = torch.concat([output1, br_activity], dim=0)
    # network.analysis.plot_firing_rates(output, population='L23e')

    model_activations = network.read_out(br_activity, mode='classification')
    br_responses = torch.argmax(model_activations, dim=1)

    return br_responses.sum().item() / len(br_responses)

def run_adaptation(network, nr_trials):

    adapter_stims, _, br_stims = get_data_batch(batch_size=nr_trials, stim_strength=20.)

    output1 = network.run({'bottom_up': torch.zeros_like(adapter_stims)},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level)
    output2 = network.run({'bottom_up': adapter_stims},
                          sim_time=1.5, input_window=[0.0, 1.5], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    output3 = network.run({'bottom_up': torch.zeros_like(adapter_stims)},
                          sim_time=1.0, input_window=[0.0, 1.0], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    br_activity = network.run({'bottom_up': br_stims},
                          sim_time=1.0, input_window=[0.0, 0.75], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    # output = torch.concat([output1, output2, output3, br_activity], dim=0)
    # network.analysis.plot_firing_rates(output, population='L23e')
    #
    # for i in range(nr_trials):
    #     plt.plot(output[:, i, 16])
    #     plt.plot(output[:, i, 24])
    #     plt.show()

    model_activations = network.read_out(br_activity, mode='classification')

    return compute_congruency(adapter_stims, model_activations)

def run_weak_perception(network, nr_trials):

    _, weak_stims, br_stims = get_data_batch(batch_size=nr_trials, img_strength=5.)

    output1 = network.run({'bottom_up': torch.zeros_like(weak_stims)},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level)
    output2 = network.run({'bottom_up': weak_stims},
                          sim_time=1.0, input_window=[0.0, 0.5], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    br_activity = network.run({'bottom_up': br_stims},
                          sim_time=1.0, input_window=[0.0, 0.75], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    # output = torch.concat([output1, output2, br_activity], dim=0)
    # network.analysis.plot_firing_rates(output, population='L23e')

    # for i in range(nr_trials):
    #     plt.plot(output[:, i, 16])
    #     plt.plot(output[:, i, 24])
    #     plt.show()

    model_activations = network.read_out(br_activity, mode='classification')

    return compute_congruency(weak_stims, model_activations)

def run_adaptation_and_weak_perception(network, nr_trials):

    adapter_stims, weak_stims, br_stims = get_data_batch(batch_size=nr_trials, stim_strength=20., img_strength=5.)

    output1 = network.run({'bottom_up': torch.zeros_like(adapter_stims)},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level)
    output2 = network.run({'bottom_up': adapter_stims},
                          sim_time=1.5, input_window=[0.0, 1.5], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    output3 = network.run({'bottom_up': weak_stims},
                          sim_time=1.0, input_window=[0.0, 0.5], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    br_activity = network.run({'bottom_up': br_stims},
                          sim_time=1.0, input_window=[0.0, 0.75], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    # output = torch.concat([output1, output2, output3, br_activity], dim=0)
    # network.analysis.plot_firing_rates(output, population='L23e')

    model_activations = network.read_out(br_activity, mode='classification')

    return compute_congruency(adapter_stims, model_activations)

def run_exp2_adaptation(network, nr_trials):

    adapter_stims, _, br_stims = get_data_batch(batch_size=nr_trials, stim_strength=20.)

    output1 = network.run({'bottom_up': torch.zeros_like(adapter_stims)},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level)
    output2 = network.run({'bottom_up': adapter_stims},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    output3 = network.run({'bottom_up': torch.zeros_like(adapter_stims)},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    br_activity = network.run({'bottom_up': br_stims},
                          sim_time=3.0, input_window=[0.0, 0.75], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    output = torch.concat([output1, output2, output3, br_activity], dim=0)
    network.analysis.plot_firing_rates(output, population='L23e')

    model_activations = network.read_out(br_activity, mode='classification')

    return compute_congruency(adapter_stims, model_activations)

def run_adaptation_and_imagery(network, nr_trials):

    adapter_stims, imagery_input, br_stims = get_data_batch(batch_size=nr_trials, stim_strength=20., img_strength=5.)

    output1 = network.run({'bottom_up': torch.zeros_like(adapter_stims),
                           'top_down': torch.zeros_like(imagery_input)},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level)
    output2 = network.run({'bottom_up': adapter_stims},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    output3 = network.run({'top_down': imagery_input},
                          sim_time=3.0, input_window=[0.0, 3.0], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    br_activity = network.run({'bottom_up': br_stims},
                          sim_time=3.0, input_window=[0.0, 0.75], stochastic=noise, noise_amplitude=noise_level, reset_state=False)
    output = torch.concat([output1, output2, output3, br_activity], dim=0)
    network.analysis.plot_firing_rates(output, population='L23e')

    model_activations = network.read_out(br_activity, mode='classification')

    return compute_congruency(adapter_stims, model_activations)

def run_conditions(network, nr_trials):

    passive_br_dominance = run_passive_br(network, nr_trials)

    adap_congruency = run_adaptation(network, nr_trials)
    weak_congruency = run_weak_perception(network, nr_trials)
    adap_weak_congruency = run_adaptation_and_weak_perception(network, nr_trials)
    adap_exp2_congruency = run_exp2_adaptation(network, nr_trials)
    adap_imagery_congruency = run_adaptation_and_imagery(network, nr_trials)

    print(
        passive_br_dominance,
        weak_congruency,
        adap_congruency,
        adap_weak_congruency,
        adap_exp2_congruency,
        adap_imagery_congruency
    )



if __name__ == '__main__':

    noise = True
    noise_level = 1.5
    adap = 1.5

    network_path = models_path('br_img', 'br_img_1.pt')
    network = BrainNetwork.load(network_path)

    network.adaptation_strength_full = torch.tile(torch.tensor([adap, 0.0]), (1, 8))
    run_conditions(network, nr_trials=4)

    network.analysis.visualize_weights(conn_name='input_top_down_v1')
