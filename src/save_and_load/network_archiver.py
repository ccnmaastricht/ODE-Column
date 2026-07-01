from datetime import datetime

from src.structure.connection import Connection
from src.utils.save_and_load import load_config



class NetworkArchiver:

    def __init__(self, network):

        self.network = network

    def export_architecture(self):
        """
        Returns an architecture dict of all areas, connections and
        output_connections that make up the network.
        """
        architecture = {"areas": [],
                        "connections": [],
                        "output_connections": []}

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
                "trainable": conn.trainable}

            architecture["connections"].append(conn_dict)

        for conn in self.network.output_connections.values():

            architecture["output_connections"].append({
                "conn_type": conn.conn_type,
                "source_id": conn.source_id,
                "target_id": conn.target_id,
                "source_size": conn.source_size,
                "target_size": conn.target_size,
                "trainable": conn.trainable})

        return architecture

    def import_architecture(self, architecture):
        """
        Import architecture and rebuild all areas and connections.
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
                conn["source_size"],
                conn["target_size"],
                initialize_weights_and_mask=True)

            self.network.connections[connection.get_name()] = connection

        for conn in architecture["output_connections"]:

            connection = Connection(
                conn["conn_type"],
                conn["source_id"],
                conn["target_id"],
                conn["trainable"],
                conn["source_size"],
                conn["target_size"],
                initialize_weights_and_mask=True)

            self.network.output_connections[connection.get_name()] = connection

    def create_checkpoint(self):
        """
        Create the checkpoint for saving the current network: store
        the architecture, weights and params, framework version and
        current date.
        """
        return {"architecture": self.export_architecture(),
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
        Reinstates the network from the saved checkpoint. If config_paths are specified,
        overwrite the saved parameters with the parameters from the config file(s).
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

        network.archive.import_architecture(checkpoint["architecture"])
        network.finalize()

        network.load_state_dict(checkpoint["state_dict"])

        return network
