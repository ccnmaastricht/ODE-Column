import torch
from pathlib import Path

from src.utils.save_and_load import load_config

from src.structure.brain_area import BrainArea
from src.structure.connection import Connection

from src.dynamics.network_dynamics import NetworkDynamics
from src.simulation.network_simulator import NetworkSimulator

from src.analysis.network_analyzer import NetworkAnalyzer
from src.analysis.network_readout import NetworkReadout
from src.save_and_load.network_archiver import NetworkArchiver



FRAMEWORK_VERSION = '0.1.0'

DEFAULT_GENERAL_CONFIG = (
    Path(__file__).parent.parent /
    "config" /
    "general_params.toml")



class BrainNetwork(torch.nn.Module):

    """
    Network class that allows the initialization of BrainArea and Connection objects.
    Additionally, contains additional modules for network dynamics, simulation and analysis.
    """

    def __init__(self, model_params, general_params):
        super().__init__()

        self.params = {"general": general_params, "model": model_params}

        self.num_populations    = None
        self.num_columns        = None
        self.area_order         = []
        self.area_slices        = {}
        self.areas              = torch.nn.ModuleDict({})
        self.connections        = torch.nn.ModuleDict({})
        self.output_connections = torch.nn.ModuleDict({})

        self._initialize_general_parameters(general_params)
        self._initialize_additional_modules()

        self.network_is_finalized = False
        self.framework_version = FRAMEWORK_VERSION

    @classmethod
    def from_toml(cls, model_config_path, general_config_path=None):
        """
        Initialize a BrainNetwork object from .toml config files.
        """
        model_params, general_params = cls._load_params_from_configs(
            model_config_path,
            general_config_path)

        return cls(
            model_params=model_params,
            general_params=general_params)

    @staticmethod
    def _load_params_from_configs(model_config_path, general_config_path):
        """
        Load model-specific and general parameters from their respective
        config files.
        """
        if general_config_path is None:
            general_config_path = DEFAULT_GENERAL_CONFIG

        model_params = load_config(model_config_path)
        general_params = load_config(general_config_path)

        return model_params, general_params

    @classmethod
    def _from_checkpoint_DEAD(cls, checkpoint, model_config_path=None, general_config_path=None):
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

        network = cls(model_params, general_params)

        network.archive.import_architecture(checkpoint["architecture"])
        network.finalize()

        network.load_state_dict(checkpoint["state_dict"])

        return network

    def _initialize_general_parameters(self, params):
        """
        Initialize general parameters that apply for the entire network.
        """
        # Background drive
        bg_drive = torch.tensor(params['background_drive'], dtype=torch.float32)
        self.register_buffer("background_drive", bg_drive.unsqueeze(0))  # add extra dim

        # Firing rate functionality
        fr_params = params['firing_rate_params']
        self.register_buffer("gain", torch.tensor(fr_params['gain'], dtype=torch.float32))
        self.register_buffer("threshold", torch.tensor(fr_params['threshold'], dtype=torch.float32))
        self.register_buffer("noise_factor", torch.tensor(fr_params['noise_factor'], dtype=torch.float32))

        # Time constants
        time_constants = params['time_constants']
        self.register_buffer("synapse_time_constant", torch.tensor(time_constants['synapse'], dtype=torch.float32))
        self.register_buffer("membrane_time_constant", torch.tensor(time_constants['membrane'], dtype=torch.float32))
        self.register_buffer("adapt_time_constant", torch.tensor(time_constants['adaptation'], dtype=torch.float32))

        # Membrane resistance and adaptation strength
        resistance = time_constants['membrane'] / params['capacitance']
        self.register_buffer("resistance", torch.tensor(resistance, dtype=torch.float32))
        self.register_buffer("adaptation_strength", torch.tensor(
            self.params['model']['adaptation_strength'], dtype=torch.float32))

    def _initialize_additional_modules(self):
        """
        Initialize NetworkDynamics(), NetworkSimulator(), NetworkReadout()
        and NetworkAnalyzer().
        """
        self.dynamics   = NetworkDynamics(self)
        self.simulator  = NetworkSimulator(self)
        self.readout    = NetworkReadout(self)
        self.analysis   = NetworkAnalyzer(self)
        self.archive    = NetworkArchiver(self)

    def _get_area(self, area_id):
        """
        Returns the Area object from the self.areas dict.
        """
        area_id = area_id.lower()
        assert area_id in self.areas.keys(), f"Area '{area_id}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."

        return self.areas[area_id]

    def add_area(self,
                 area_name,
                 size,
                 unique_id=None,
                 intrinsic_trainable=False,
                 background_trainable=False):
        """
        Initialize the specified area and its recurrent and background connections.

        Params:
        area_name (str):                The name of the to-be-modeled area, as specified in the .toml file (e.g. 'v1', 'v2', etc).
        size (int):                     The number of columns of the area.
        unique_id (str):                An optional user-specified id for the area. Useful when the network should contain more
                                        area modules with the same area configurations.
        intrinsic_trainable (bool):     If True, the recurrent, column-intrinsic connections can be updated during training.
        background_trainable (bool):    If True, the background connections can be updated during training.
        """
        area_name = area_name.lower()
        assert area_name in self.params['general']['population_size'], f"Population sizes of '{area_name}' not found in .toml file. "
        assert area_name in self.params['general']['background_synapse_counts'], f"Background synapse counts of '{area_name}' not found in .toml file. "

        if unique_id is None:
            unique_id = area_name

        area = BrainArea(self.params['general'], area_name, size, unique_id)
        self.areas[unique_id] = area

        # Add recurrent connectivity and background connectivity as connections
        self.add_recurrent_connection(area, unique_id, intrinsic_trainable)
        self.add_background_connection(area, unique_id, background_trainable)

    def add_recurrent_connection(self,
                                 area,
                                 unique_area_id,
                                 trainable=True):

        recurrent_connection = Connection('recurrent', unique_area_id, unique_area_id, trainable)
        recurrent_connection.initialize_recurrent_weights(area)
        self.connections[recurrent_connection.get_name()] = recurrent_connection

    def add_background_connection(self,
                                  area,
                                  unique_area_id,
                                  trainable=True):

        background_connection = Connection('background', 'background', unique_area_id, trainable)
        background_connection.initialize_background_weights(area)
        self.connections[background_connection.get_name()] = background_connection

    def add_feedforward_connection(self,
                                   source,
                                   target,
                                   trainable=True,
                                   std=0.1,
                                   scale=1.0):

        source_area = self._get_area(source)
        target_area = self._get_area(target)

        connection = Connection('feedforward', source, target, trainable)
        connection.initialize_feedforward_weights(self.params, source_area, target_area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_feedback_connection(self,
                                source,
                                target,
                                trainable=True,
                                std=0.1,
                                scale=1.0):

        source_area = self._get_area(source)
        target_area = self._get_area(target)

        connection = Connection('feedback', source, target, trainable)
        connection.initialize_feedback_weights(self.params, source_area, target_area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_lateral_connection(self,
                               area_name,
                               trainable=True,
                               std=0.1,
                               scale=1.0):

        area = self._get_area(area_name)

        connection = Connection('lateral', area_name, area_name, trainable)
        connection.initialize_lateral_weights(self.params, area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_input_connection(self,
                             target_area,
                             input_size,
                             trainable=True,
                             unique_id=None,
                             std=0.1,
                             scale=1.0):

        area = self._get_area(target_area)

        input_name = 'input'
        if unique_id is not None:
            input_name = unique_id

        connection = Connection('input', input_name, target_area, trainable)
        connection.initialize_input_weights(self.params, input_size, area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_output_connection(self,
                              source_area,
                              trainable=True,
                              unique_id=None,
                              std=0.0,
                              scale=1.0):

        area = self._get_area(source_area)

        output_name = 'output'
        if unique_id is not None:
            output_name = unique_id

        connection = Connection('output', source_area, output_name, trainable)
        connection.initialize_output_weights(self.params, area, std, scale)
        self.output_connections[connection.get_name()] = connection

    def finalize(self):
        """
        Finalizes the network after initializing all areas and connections by setting
        the total number of populations and columns, and setting slices to index the
        activity of each area. Also extends adaptation strength to entire network.
        """
        if not self.network_is_finalized:

            self.num_populations = sum(area.num_populations for area in self.areas.values())
            self.num_columns = self.num_populations // 8

            # Area order in state and slices per area
            idx = 0
            for area_id, area in self.areas.items():
                self.area_order.append(area_id)
                self.area_slices[area_id] = slice(idx, idx + area.num_populations)
                idx += area.num_populations

            # Extend adaptation strength tensor to cover the entire network
            self.register_buffer("adaptation_strength_full", torch.tile(self.adaptation_strength,(self.num_columns,)))

            self.network_is_finalized = True

    def constrain_weights(self):
        """
        Constrain all connection weights to not use any illegal connections.
        """
        all_connections = (list(self.connections.values())
                           + list(self.output_connections.values()))

        for connection in all_connections:
            connection.constrain(self.areas.keys())

    def run(self, ext_input=None, input_window=None, adjoint=False, stochastic=False, device="cpu"):
        """
        Delegates running the network to the simulation engine.
        """
        return self.simulator.run(
            ext_input=ext_input,
            input_window=input_window,
            adjoint=adjoint,
            stochastic=stochastic,
            device=device)

    def get_firing_rates(self, raw_state, area=None, return_as_np_array=True):
        """
        Computes the firing rate from the raw state (= [membrane_potential, adaptation])
        Returns as np.array unless specified otherwise.
        """
        return self.readout.get_firing_rates(
            raw_state=raw_state,
            area=area,
            return_as_np_array=return_as_np_array)

    def read_out(self, raw_output, mode, sum_per_col=True):
        """
        Reads model output from raw output; converts to firing rates and slices
        only the area(s) that are identified as output sources.
        Either returns entire trajectory (mode='trajectory') or average of
        last x time steps (mode='classification'), i.e. trajectory-based vs
        classification-based training procedure.
        """
        return self.readout.read_out(
            raw_output=raw_output,
            mode=mode,
            sum_per_col=sum_per_col)

    def save(self, path):
        """
        Save the current network as a checkpoint, with its learned weights
        and additional parameters.
        """
        checkpoint = self.archive.create_checkpoint()
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, model_config_path=None, general_config_path=None):
        """
        Load a saved network checkpoint and reinstate the network with its
        saved weights and parameters.
        """
        checkpoint = torch.load(path, weights_only=False)

        return NetworkArchiver.restore_checkpoint(
            cls,
            checkpoint,
            model_config_path,
            general_config_path)