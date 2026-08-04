from datetime import datetime

from src.structure.connection import Connection
from src.utils.save_and_load import load_config


class NetworkArchiver:
    """
    Handles network architecture serialization, checkpoint creation, and state
    restoration.

    Args:
        network (BrainNetwork): Target brain network module.

    Attributes:
        network (BrainNetwork): Reference to the target brain network.
    """

    def __init__(self, network):

        self.network = network

    def _export_architecture(self):
        """
        Export a structured dictionary detailing all brain areas and connections
        that comprise the network architecture.

        Returns:
            dict[str, list[dict]]: Architecture dictionary containing specifications
                for area sizes, connectivity sources, targets, and trainability flags.
        """
        architecture = {"areas": [],
                        "connections": []}

        for area_id, area in self.network.areas.items():

            architecture["areas"].append({
                "id": area_id,
                "area_name": area.area_name,
                "size": area.num_columns})

        for conn in self.network.connections.values():

            conn_dict = {
                "conn_type": conn.conn_type,
                "source_id": conn.source_id,
                "target_id": conn.target_id,
                "source_size": conn.source_size,
                "target_size": conn.target_size,
                "trainable": conn.trainable,
                "dales_law_constraint": conn.dales_law_constraint}

            architecture["connections"].append(conn_dict)

        return architecture

    def _import_architecture(self, architecture):
        """
        Reconstruct all network brain areas and connections from a
        serialized architecture dictionary.

        Args:
            architecture (dict[str, list[dict]]): Architecture dictionary exported
                via `_export_architecture`.
        """
        for area in architecture["areas"]:

            self.network.add_area(
                area_name=area["area_name"],
                size=area["size"],
                unique_id=area["id"],
                initialize_recurrent_and_background=False)

        for conn in architecture["connections"]:

            connection = Connection(
                conn["conn_type"],
                conn["source_id"],
                conn["target_id"],
                conn["trainable"],
                conn["dales_law_constraint"],
                conn["source_size"],
                conn["target_size"],
                initialize_weights_and_mask=True)

            self.network.connections[connection.get_name()] = connection

    def create_checkpoint(self):
        """
        Construct a comprehensive state checkpoint dictionary containing serialized
        architecture layout, PyTorch state dictionary weights, model and general
        parameters, framework version, and timestamp.

        Returns:
            dict[str, Any]: Complete model checkpoint payload dictionary.
        """
        return {"architecture": self._export_architecture(),
                "state_dict": self.network.state_dict(),

                "model_params": self.network.params['model'],
                "general_params": self.network.params['general'],

                "framework_version": self.network.framework_version,
                "date": str(datetime.today().strftime('%Y-%m-%d'))}

    @classmethod
    def restore_checkpoint(
            cls,
            network_cls,
            checkpoint,
            model_config_path=None,
            general_config_path=None):
        """
        Restore a complete BrainNetwork model instance from a saved checkpoint,
        optionally overriding parameters with external configuration files.

        Args:
            network_cls (type[BrainNetwork]): Class type reference to instantiate the
                network.
            checkpoint (dict[str, Any]): Checkpoint dictionary loaded from disk.
            model_config_path (str | Path | None, optional): Path to model configuration
                file to override stored parameters. Defaults to None.
            general_config_path (str | Path | None, optional): Path to general configuration
                file to override stored parameters. Defaults to None.

        Returns:
            BrainNetwork: Fully restored and finalized network instance with loaded
                weights and buffers.
        """
        # Decide which parameters to use
        if model_config_path is None:
            model_params = checkpoint["model_params"]
        else:
            model_params = load_config(model_config_path)
            print("Loading network with overridden model parameters instead of saved model parameters.")

        if general_config_path is None:
            general_params = checkpoint["general_params"]
        else:
            general_params = load_config(general_config_path)
            print("Loading network with overridden general parameters instead of saved general parameters.")

        network = network_cls(model_params, general_params)

        network.archive._import_architecture(checkpoint["architecture"])
        network.finalize()

        network.load_state_dict(checkpoint["state_dict"])

        return network
