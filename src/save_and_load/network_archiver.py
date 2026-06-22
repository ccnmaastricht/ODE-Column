from datetime import datetime

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
                "trainable": conn.trainable}

            if conn.conn_type == 'input':
                conn_dict["input_size"] = conn.size_input

            architecture["connections"].append(conn_dict)

        for conn in self.network.output_connections.values():

            architecture["output_connections"].append({
                "conn_type": conn.conn_type,
                "source_id": conn.source_id,
                "target_id": conn.target_id,
                "trainable": conn.trainable})

        return architecture

    def import_architecture(self, architecture):
        """
        Import architecture and rebuild all areas and connections.
        """
        for area in architecture["areas"]:

            self.network.add_area(area_name=area["area_name"],
                          size=area["size"],
                          unique_id=area["id"])

        for conn in architecture["connections"]:

            if conn["conn_type"] == "recurrent":
                self.network.add_recurrent_connection(
                    area=self.network.areas[conn["target_id"]],
                    unique_area_id=conn["source_id"],
                    trainable=conn["trainable"])

            elif conn["conn_type"] == "background":
                self.network.add_background_connection(
                    area=self.network.areas[conn["target_id"]],
                    unique_area_id=conn["target_id"],
                    trainable=conn["trainable"])

            elif conn["conn_type"] == "feedforward":
                self.network.add_feedforward_connection(
                    source=conn["source_id"],
                    target=conn["target_id"],
                    trainable=conn["trainable"])

            elif conn["conn_type"] == "feedback":
                self.network.add_feedback_connection(
                    source=conn["source_id"],
                    target=conn["target_id"],
                    trainable=conn["trainable"])

            elif conn["conn_type"] == "lateral":
                self.network.add_lateral_connection(
                    area_name=conn["source_id"],
                    trainable=conn["trainable"])

            elif conn["conn_type"] == "input":
                self.network.add_input_connection(
                    target_area=conn["target_id"],
                    input_size=conn["input_size"],
                    unique_id=conn["source_id"],
                    trainable=conn["trainable"])

        for conn in architecture["output_connections"]:

            self.network.add_output_connection(
                source_area=conn["source_id"],
                unique_id=conn["target_id"],
                trainable=conn["trainable"])

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
