import matplotlib.pyplot as plt


class NetworkAnalyzer:
    """
    Provides visualization and inspection tools for network firing rates, synaptic
    connectivity weights, and structural connection properties.

    Args:
        network (BrainNetwork): Target brain network module.

    Attributes:
        network (BrainNetwork): Reference to the target brain network.
        layer_labels (list[str]): Names of laminar populations (`['L23e', 'L23i', ...]`).
    """

    def __init__(self, network):

        self.network = network
        self.layer_labels = network.readout.layer_labels

    def plot_firing_rates(self, raw_state, sample=None, area=None, column=None, population=None):
        """
        Plot population firing rate trajectories over time for specified samples, areas,
        columns, and laminar populations using Matplotlib.

        Args:
            raw_state (torch.Tensor): Simulation output tensor containing raw state
                variables (membrane potential and adaptation), shape
                `(time_steps, batch_size, 2 * total_populations)`.
            sample (int | list[int] | None, optional): Specific sample index or list of
                sample indices to plot. Defaults to None (all samples).
            area (str | None, optional): Area name to slice. Defaults to None (all areas).
            column (int | list[int] | None, optional): Column index or list of column
                indices to plot. Defaults to None (all columns).
            population (int | str | list | None, optional): Laminar population index
                or name (e.g., 'L23e') to plot. Defaults to None (all populations).
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
        Extract synaptic weight matrices for specified or all network connections,
        returning raw or rectifying-constrained weights as PyTorch Tensors or NumPy
        arrays.

        Args:
            conn_name (str | None, optional): Name of a specific connection to extract.
                Defaults to None (returns dictionary of all connections).
            get_constrained_weights (bool, optional): Whether to apply non-negative
                excitatory and non-positive inhibitory weight constraints (`conn.W`).
                Defaults to True.
            return_as_np (bool, optional): Whether to convert weights to NumPy arrays.
                Defaults to True.

        Returns:
            dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor: Dictionary
                mapping connection names to weight matrices, or a single weight matrix
                if `conn_name` is specified.
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

    def visualize_weights(self, conn_name=None, get_constrained_weights=True):
        """
        Visualize connection weight matrices as 2D heatmaps using Matplotlib's viridis
        color map.

        Args:
            conn_name (str | None, optional): Specific connection name to visualize.
                Defaults to None (plots heatmaps for all connections).
            get_constrained_weights (bool, optional): Whether to plot constrained
                weights (`conn.W`) instead of raw weights (`conn.weights`). Defaults
                to False.
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
        Print a formatted tabular summary of all initialized network connections,
        including source, target, tensor shape, trainability, and connection type.
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
