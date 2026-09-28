from src.brain_network import BrainNetwork
from src.utils.paths import models_path



if __name__ == '__main__':

    network_path = models_path('br_img', 'br_img_1.pt')
    network = BrainNetwork.load(network_path)

    network.analysis.visualize_weights()

    weights = network.analysis.get_weights(conn_name='input_top_down_v1')

