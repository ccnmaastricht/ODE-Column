from src.brain_network import BrainNetwork
from src.utils.paths import models_path



if __name__ == '__main__':

    network_path = models_path('memory', f'memory_1.pt')
    history_path = models_path('memory', f'memory.pt')

    network = BrainNetwork.load(network_path)

    input_weights = network.analysis.get_weights(conn_name='input_input_v1')
    ff_weights = network.analysis.get_weights(conn_name='feedforward_v1_v2')
    lateral_weights = network.analysis.get_weights(conn_name='lateral_v1_v1')
    stop = 0
