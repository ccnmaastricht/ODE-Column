import matplotlib.pyplot as plt



class NetworkAnalyzer:

    def __init__(self, network):

        self.network = network
        self.layer_labels = network.readout.layer_labels

    def plot_firing_rates(self, raw_state, sample=None, area=None, column=None, population=None):
        """
        Plot firing rates.
        """
        firing_rates, sample_list, area_list, column_list, pop_list = self.network.readout.get_firing_rates(
            raw_state, sample, area, column, population, return_as_dict=True)

        for sample_ in range(len(sample_list)):

            for area_ in area_list:
                for col in range(len(column_list[area_])):
                    for pop in range(len(pop_list)):

                        pop_name = self.layer_labels[pop_list[pop]]
                        label = f'{area_}_col{column_list[area_][col]}_{pop_name}'
                        plt.plot(firing_rates[area_][:, sample_, col, pop], label=label)

            plt.legend()
            plt.title(f'Sample {sample_list[sample_]}')
            plt.show()

    def get_weights(self, conn_name=None, get_constrained_weights=True, return_as_np=True):
        """
        Returns a dictionary of all connection in the network with their
        weight matrices as values. If param 'conn_name' is specified, the
        function returns only the weights of this connection.
        """
        if get_constrained_weights:
            conn_dict = {name : conn.W for name, conn in self.network.connections.items()}
        else:
            conn_dict = {name : conn.weights for name, conn in self.network.connections.items()}

        if return_as_np:
            conn_dict = {name : weights.detach().cpu().numpy() for name, weights in conn_dict.items()}

        if conn_name is not None:
            return conn_dict[conn_name]

        return conn_dict

    def visualize_weights(self, conn_name=None, get_constrained_weights=False):
        """
        Visualize the weights as a heatmap.
        """
        conn_dict = self.get_weights(conn_name, get_constrained_weights)

        for name, weights in conn_dict.items():
            fig, ax = plt.subplots()

            heatmap = ax.imshow(weights, cmap="viridis", interpolation="nearest")
            fig.colorbar(heatmap, ax=ax)

            ax.set_title(name)
            ax.set_xlabel("Source")
            ax.set_ylabel("Target")

            plt.show()

    def summarize_connections(self):
        """
        Prints a summary of all initialized network connections.
        """
        header = (
            f"{'Name':35}"
            f"{'Source':15}"
            f"{'Target':15}"
            f"{'Shape':15}"
            f"{'Trainable':10}"
            f"{'Type':12}"
        )
        print(header)
        print("-" * len(header))

        for name, conn in self.network.connections.items():
            print(
                f"{name:35}"
                f"{str(conn.source_id):15}"
                f"{str(conn.target_id):15}"
                f"{str((conn.target_size, conn.source_size)):15}"
                f"{str(conn.trainable):10}"
                f"{str(conn.conn_type):12}")
