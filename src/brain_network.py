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
    Additionally, contains modules for network dynamics, simulation, analysis readout
    and archiving.
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
        Initialize NetworkDynamics(), NetworkSimulator(), NetworkReadout(),
        NetworkAnalyzer() and NetworkArchiver().
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

    def _initialize_default_connection(
            self,
            connection_type,
            source,
            target,
            initializer,
            *initializer_args,
            trainable=True,
            output=False):
        """
        Initialize the specified default connection as a Connection object and store it
        in the appropriate dict.

        Params:
        connection_type (str):          Specified connection type. Appropriate types are 'input', 'output',
                                        'recurrent', 'background', 'feedforward', 'feedback', 'lateral'.
        source (str):                   ID of source.
        target (str):                   ID of target.
        initializer (func):             Specified function to be used to initialize the weights. Should
                                        return (weights, mask).
        *initializer_args:              Params passed to initializer func. See below for more information.
        trainable (bool):               If True, the feedforward weights should be updated during training.
                                        If False, the weights should remain static.
        output (bool):                  If True, the connection should be added to the output_connections dict
                                        instead of the regular connections dict.
        ================================
        *initializer_args are different across the various connection types. Find the exhaustive list of
        possible params below:
        params (dict):                  BrainNetwork's parameters, both general and model-specific.
        source_area (BrainArea):        Source BrainArea object for connection types 'output', 'feedforward',
                                        and 'feedback'.
        target_area (BrainArea):        Target BrainArea object for connection types 'input', 'feedforward',
                                        and 'feedback'.
        area (BrainArea):               Source *and* target BrainArea object for connection types 'recurrent',
                                        'background' and 'lateral'.
        size_input (int):               Size of the input for connection type 'input'.
        receptive_field_size (int):     Size of the receptive fields that constrain the input the target area
                                        receives from the source area. Leave as None if the source and target
                                        area should be fully connected.
        stride (int):                   Stride (step size) of the receptive field window.
        grid_organization (bool):       If True, the receptive field connectivity between the source and target
                                        area will assume a two-dimensional (i.e. grid) organization.
                                        If False, connectivity will assume a one-dimensional organization.
        std (float):                    Standard deviation of the randomly initialized weights. Mean will be
                                        determined from the user-specified .toml file.
        scale (float):                  Scale of the initialized weights.
        """
        connection = Connection(
            connection_type,
            source,
            target,
            trainable)

        weights, mask = initializer(connection, *initializer_args)

        connection.set_sizes(weights.shape[1], weights.shape[0])
        connection.set_weights_and_mask(weights, mask)

        registry = (self.output_connections
                    if output
                    else self.connections)

        registry[connection.get_name()] = connection

    def add_area(
            self,
            area_name,
            size,
            unique_id=None,
            initialize_recurrent_and_background=True,
            intrinsic_trainable=False,
            background_trainable=False):
        """
        Initialize the specified area and its recurrent and background connections.

        Params:
        area_name (str):                The name of the added area, as specified in the .toml file (e.g. 'v1', 'v2', etc).
        size (int):                     The number of columns of the area.
        unique_id (str):                An optional user-specified id for the area. Useful when the network should contain more
                                        area modules with the same area configurations.
        initialize_recurrent_and        If False, adding the area will not automatically add recurrent and background
            _background (bool):         connections.
        intrinsic_trainable (bool):     If True, the recurrent (column-intrinsic) connections can be updated during training.
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
        if initialize_recurrent_and_background:
            self.add_recurrent_connection(area, unique_id, intrinsic_trainable)
            self.add_background_connection(area, unique_id, background_trainable)

    def add_recurrent_connection(
            self,
            area,
            unique_area_id,
            trainable=True):
        """
        Add recurrent (column-intrinsic) connections to the network architecture.

        Params:
        area (str):                     Area for which recurrent connections should be initialized.
        unique_area_id (str):           User-specified id for the area.
        trainable (bool):               If True, weights should be updated during training.
        ================================
        See _initialize_default_connection() for more initialization options.
        """
        self._initialize_default_connection(
            'recurrent',
            unique_area_id,
            unique_area_id,
            Connection.initialize_recurrent_weights,
            area,
            trainable=trainable)

    def add_background_connection(
            self,
            area,
            unique_area_id,
            trainable=True):
        """
        Add background connections to the network architecture.

        Params:
        area (str):                     Area for which background connections should be initialized.
        unique_area_id (str):           User-specified id for the area.
        trainable (bool):               If True, weights should be updated during training.
        ================================
        See _initialize_default_connection() for more initialization options.
        """
        self._initialize_default_connection(
            'background',
            'background',
            unique_area_id,
            Connection.initialize_background_weights,
            area,
            trainable=trainable)

    def add_feedforward_connection(
            self,
            source,
            target,
            trainable=True,
            receptive_field_size=None,
            stride=1,
            grid_organization=False,
            std=0.1,
            scale=1.0):
        """
        Add a feedforward connection to the network architecture.

        Params:
        source (str):                   Source area that the feedforward connection originates from.
        target (str):                   Target area of the feedforward connection.
        trainable (bool):               If True, weights should be updated during training.
        ================================
        See _initialize_default_connection() for more initialization options.
        """
        source_area = self._get_area(source)
        target_area = self._get_area(target)

        self._initialize_default_connection(
            'feedforward',
            source,
            target,
            Connection.initialize_feedforward_weights,
            self.params,
            source_area,
            target_area,
            receptive_field_size,
            stride,
            grid_organization,
            std,
            scale,
            trainable=trainable)

    def add_feedback_connection(
            self,
            source,
            target,
            trainable=True,
            receptive_field_size=None,
            stride=1,
            grid_organization=False,
            std=0.1,
            scale=1.0):
        """
        Add a feedback connection to the network architecture.

        Params:
        source (str):                   Source area that the feedback connection originates from.
        target (str):                   Target area of the feedback connection.
        trainable (bool):               If True, weights should be updated during training.
        ================================
        See _initialize_default_connection() for more initialization options.
        """
        source_area = self._get_area(source)
        target_area = self._get_area(target)

        self._initialize_default_connection(
            'feedback',
            source,
            target,
            Connection.initialize_feedback_weights,
            self.params,
            source_area,
            target_area,
            receptive_field_size,
            stride,
            grid_organization,
            std,
            scale,
            trainable=trainable)

    def add_lateral_connection(
            self,
            area_name,
            trainable=True,
            receptive_field_size=None,
            stride=1,
            grid_organization=False,
            std=0.1,
            scale=1.0):
        """
        Add lateral connections to the network architecture.

        Params:
        area_name (str):                Area for which lateral connections should be initialized.
        trainable (bool):               If True, weights should be updated during training.
        ================================
        See _initialize_default_connection() for more initialization options.
        """
        area = self._get_area(area_name)

        self._initialize_default_connection(
            'lateral',
            area_name,
            area_name,
            Connection.initialize_lateral_weights,
            self.params,
            area,
            receptive_field_size,
            stride,
            grid_organization,
            std,
            scale,
            trainable=trainable)

    def add_input_connection(
            self,
            target_area,
            input_size,
            unique_id=None,
            trainable=True,
            receptive_field_size=None,
            stride=1,
            grid_organization=False,
            std=0.1,
            scale=1.0):
        """
        Add an input connection to the network architecture.

        Params:
        target_area (str):              Target area of the input connection.
        input_size (int):               Size of input. If input is two-dimensional, please pass flattened size
                                        (and set grid_organization=True).
        unique_id (str):                An optional user-specified id for the input. Useful when the network
                                        should contain more than one input connection. If using, make sure there
                                        is a corresponding mask and init in the .toml file with the same id.
        trainable (bool):               If True, weights should be updated during training.
        ================================
        See _initialize_default_connection() for more initialization options.
        """
        area = self._get_area(target_area)
        input_name = unique_id or 'input'

        self._initialize_default_connection(
            'input',
            input_name,
            target_area,
            Connection.initialize_input_weights,
            self.params,
            input_size,
            area,
            receptive_field_size,
            stride,
            grid_organization,
            std,
            scale,
            trainable=trainable)

    def add_output_connection(
            self,
            source_area,
            unique_id=None,
            trainable=False,
            std=0.0,
            scale=1.0):
        """
        Add an output connection to the network architecture.

        Params:
        source_area (str):              Source area of the output connection.
        unique_id (str):                An optional user-specified id for the output. Useful when the network
                                        should contain more than one output connection. If using, make sure there
                                        is a corresponding mask and init in the .toml file with the same id.
        trainable (bool):               If True, weights should be updated during training.
        ================================
        See _initialize_default_connection() for more initialization options.
        """
        area = self._get_area(source_area)
        output_name = unique_id or 'output'

        self._initialize_default_connection(
            'output',
            source_area,
            output_name,
            Connection.initialize_output_weights,
            self.params,
            area,
            std,
            scale,
            trainable=trainable,
            output=True)

    def add_custom_connection(
            self,
            connection_name,
            source,
            target,
            initializer,
            trainable=True,
            is_output_connection=False,
            **initializer_args):
        """
        Add a custom connection to the network architecture with a user-specified initializer function.

        Params:
        connection_name (str):          Name of the connection, can be anything (also 'feedforward', 'input', etc.)
                                        as long as the {name}_{source}_{target} string is unique to the network.
        source (str):                   Name of the source, can refer to an already initialized network area with
                                        BrainNetwork.add_area(), or to a network-external source (i.e. 'input').
        target (str):                   Name of the source, can refer to an already initialized network area with
                                        BrainNetwork.add_area(), or to a network-external target (i.e. 'output').
        initializer (func):             A user-specified function that initializes the connection weights and mask.
                                        The function receives a dictionary with network parameters ('source', 'target',
                                        'general_params' and 'model_params' as the first argument, and any optional
                                        user-specified arguments. The function should return tensors (weights, mask)
                                        both with the shape (target, source).
        trainable (bool):               If True, weights should be updated during training.
        is_output_connection (bool):    If True, the network will handle the connection as output (i.e. read-out) and
                                        the connection itself has no influence on network dynamics. In this case,
                                        the target should not refer to a brain area.
        ** initializer_args:            Optional arguments to pass to the user-specified initializer function.
        """
        connection = Connection(
            connection_name,
            source,
            target,
            trainable)

        # If source or target are initialized brain areas, get their BrainArea object
        if source in self.areas.keys():
            source = self.areas[source]
        if target in self.areas.keys():
            target = self.areas[target]

        init_dict = {
            'source': source,
            'target': target,
            'general_params': self.params['general'],
            'model_params': self.params['model']}

        weights, mask = initializer(init_dict, **initializer_args)

        connection.set_sizes(weights.shape[1], weights.shape[0])
        connection.set_weights_and_mask(weights, mask)

        registry = (self.output_connections
                    if is_output_connection
                    else self.connections)

        registry[connection.get_name()] = connection

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

    def run(self, ext_input=None, input_window=None, adjoint=False, stochastic=False, reset_state=True, device="cpu"):
        """
        Delegates running the network to the simulation engine.
        Todo: add documentation
        reset_state=False can only be done if the network has been simulated after
        initialization and if the previous batch_size matches the current batch_size.
        """
        return self.simulator.run(
            ext_input=ext_input,
            input_window=input_window,
            adjoint=adjoint,
            stochastic=stochastic,
            reset_state=reset_state,
            device=device)

    def get_firing_rates(
            self,
            raw_state,
            sample=None,
            area=None,
            column=None,
            population=None,
            return_as_np_array=True,
            return_as_dict=False):
        """
        Computes the firing rate from the raw state (= [membrane_potential, adaptation])
        Returns as np.array unless specified otherwise.
        """
        return self.readout.get_firing_rates(
            raw_state=raw_state,
            sample=sample,
            area=area,
            column=column,
            population=population,
            return_as_np_array=return_as_np_array,
            return_as_dict=return_as_dict)

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