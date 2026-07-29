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
    Main container class for building, simulating, analyzing, and archiving networks
    composed of laminar cortical column areas and structural connections.

    Args:
        model_params (dict): Model-specific configuration parameters dictionary.
        general_params (dict): General parameter settings (time constants, gains, drive).

    Attributes:
        params (dict): Configuration dictionary containing model and general parameter blocks.
        num_populations (int | None): Total number of neuronal populations across all areas.
        num_columns (int | None): Total number of cortical columns in the network.
        area_order (list[str]): Ordered list of brain area IDs matching state vector layout.
        area_slices (dict[str, slice]): Dictionary mapping area IDs to state vector slices.
        areas (torch.nn.ModuleDict): Dictionary mapping area IDs to `BrainArea` instances.
        connections (torch.nn.ModuleDict): Dictionary mapping connection names to `Connection` instances.
        dynamics (NetworkDynamics): Bound network dynamics module.
        simulator (NetworkSimulator): Bound numerical simulation engine.
        readout (NetworkReadout): Bound signal readout and post-processing module.
        analysis (NetworkAnalyzer): Bound plotting and weight inspection module.
        archive (NetworkArchiver): Bound checkpointing and architecture serialization module.
        network_is_finalized (bool): Readiness flag set once network indexing is completed.
        framework_version (str): Installed framework version string.
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

        self._initialize_general_parameters(general_params)
        self._initialize_additional_modules()

        self.network_is_finalized = False
        self.framework_version = FRAMEWORK_VERSION

    @classmethod
    def from_toml(cls, model_config_path, general_config_path=None):
        """
        Instantiate and initialize a BrainNetwork object directly from TOML parameter configuration files.

        Args:
            model_config_path (str | Path): Path to model configuration file (`.toml`).
            general_config_path (str | Path | None, optional): Path to general parameter configuration file
                (`.toml`). Defaults to None.

        Returns:
            BrainNetwork: Newly instantiated network instance initialized with configuration parameters.
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
        Load parameter dictionaries from TOML configuration files, substituting default general parameters
        if unassigned.

        Args:
            model_config_path (str | Path): Path to model config file.
            general_config_path (str | Path | None): Path to general config file.

        Returns:
            tuple[dict, dict]: Tuple containing model parameters dictionary and general parameters dictionary.
        """
        if general_config_path is None:
            general_config_path = DEFAULT_GENERAL_CONFIG

        model_params = load_config(model_config_path)
        general_params = load_config(general_config_path)

        return model_params, general_params

    def _initialize_general_parameters(self, params):
        """
        Register global network buffers for background drive, firing rate gain/threshold/noise parameters, and time constants.

        Args:
            params (dict): General configuration dictionary.
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
        self.register_buffer("adaptation_strength", torch.tensor(params['adaptation_strength'], dtype=torch.float32))

    def _initialize_additional_modules(self):
        """
        Bind dynamics, simulator, readout, analysis, and archiver modules to this network instance.
        """
        self.dynamics   = NetworkDynamics(self)
        self.simulator  = NetworkSimulator(self)
        self.readout    = NetworkReadout(self)
        self.analysis   = NetworkAnalyzer(self)
        self.archive    = NetworkArchiver(self)

    def _get_area(self, area_id):
        """
        Retrieve a BrainArea instance from the areas dictionary by identifier string.

        Args:
            area_id (str): Name or unique ID of target area.

        Returns:
            BrainArea: Retained brain area module.

        Raises:
            AssertionError: Raised if requested area ID is not registered.
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
            trainable=True):
        """
        Instantiate a Connection object using a specified initializer function and register
        it within internal or output connection dictionaries.

        Args:
            connection_type (str): Connection type classification string.
            source (str): Source identifier string.
            target (str): Target identifier string.
            initializer (Callable): Function initializing weight and mask matrices.
            *initializer_args: Positional arguments passed to initializer function.
            trainable (bool, optional): Whether weights parameter requires grad. Defaults to True.
        """
        dales_law_constraint = True if source in self.areas.keys() else False

        connection = Connection(
            connection_type,
            source,
            target,
            trainable,
            dales_law_constraint)

        weights, mask = initializer(connection, *initializer_args)

        connection.set_sizes(weights.shape[1], weights.shape[0])
        connection.set_weights_and_mask(weights, mask)

        self.connections[connection.get_name()] = connection

    def add_area(
            self,
            area_name,
            size,
            unique_id=None,
            initialize_recurrent_and_background=True,
            intrinsic_trainable=False,
            background_trainable=False):
        """
        Add a cortical column area to the network architecture and optionally establish its
        intrinsic recurrent and background connections.

        Args:
            area_name (str): Configured area name string matching parameters in configuration
                file.
            size (int): Number of cortical columns contained within the area.
            unique_id (str | None, optional): Unique area identifier string. Defaults to None
                (uses `area_name`).
            initialize_recurrent_and_background (bool, optional): Whether to automatically
                instantiate recurrent and background connections. Defaults to True.
            intrinsic_trainable (bool, optional): Whether recurrent connection weights are trainable.
                Defaults to False.
            background_trainable (bool, optional): Whether background connection weights are trainable.
                Defaults to False.
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
        Add column-intrinsic recurrent connections to a specific brain area.

        Args:
            area (BrainArea | str): BrainArea object or area identifier string.
            unique_area_id (str): Unique area identifier string.
            trainable (bool, optional): Whether recurrent weights are trainable during optimization. Defaults to True.
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
        Add background drive connections to a specific brain area.

        Args:
            area (BrainArea | str): BrainArea object or area identifier string.
            unique_area_id (str): Unique area identifier string.
            trainable (bool, optional): Whether background weights are trainable during optimization. Defaults to True.
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
        Add inter-area feedforward connectivity between a source area and target area with optional receptive field constraints.

        Args:
            source (str): Identifier of source area.
            target (str): Identifier of target area.
            trainable (bool, optional): Whether weights are trainable during optimization. Defaults to True.
            receptive_field_size (int | None, optional): Size of spatial receptive field window. Defaults to None.
            stride (int, optional): Stride step size of receptive field window. Defaults to 1.
            grid_organization (bool, optional): Whether connectivity assumes 2D grid arrangement. Defaults to False.
            std (float, optional): Standard deviation of initialized weights. Defaults to 0.1.
            scale (float, optional): Scaling multiplier for initialized weights. Defaults to 1.0.
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
        Add inter-area feedback connectivity from a source area to a target area with optional receptive field constraints.

        Args:
            source (str): Identifier of source area.
            target (str): Identifier of target area.
            trainable (bool, optional): Whether weights are trainable during optimization. Defaults to True.
            receptive_field_size (int | None, optional): Size of spatial receptive field window. Defaults to None.
            stride (int, optional): Stride step size of receptive field window. Defaults to 1.
            grid_organization (bool, optional): Whether connectivity assumes 2D grid arrangement. Defaults to False.
            std (float, optional): Standard deviation of initialized weights. Defaults to 0.1.
            scale (float, optional): Scaling multiplier for initialized weights. Defaults to 1.0.
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
        Add inter-column lateral connectivity within an area, excluding column-intrinsic recurrent connections.

        Args:
            area_name (str): Identifier of target area.
            trainable (bool, optional): Whether weights are trainable during optimization. Defaults to True.
            receptive_field_size (int | None, optional): Size of spatial receptive field window. Defaults to None.
            stride (int, optional): Stride step size of receptive field window. Defaults to 1.
            grid_organization (bool, optional): Whether connectivity assumes 2D grid arrangement. Defaults to False.
            std (float, optional): Standard deviation of initialized weights. Defaults to 0.1.
            scale (float, optional): Scaling multiplier for initialized weights. Defaults to 1.0.
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
        Add external drive input connections targeting a specific brain area.

        Args:
            target_area (str): Identifier of target brain area receiving input.
            input_size (int): Dimension of external input vector.
            unique_id (str | None, optional): Unique ID string for input connection. Defaults to None (uses 'input').
            trainable (bool, optional): Whether weights are trainable during optimization. Defaults to True.
            receptive_field_size (int | None, optional): Size of spatial receptive field window. Defaults to None.
            stride (int, optional): Stride step size of receptive field window. Defaults to 1.
            grid_organization (bool, optional): Whether connectivity assumes 2D grid arrangement. Defaults to False.
            std (float, optional): Standard deviation of initialized weights. Defaults to 0.1.
            scale (float, optional): Scaling multiplier for initialized weights. Defaults to 1.0.
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
        Add task readout output connection reading out activity from a source brain area.

        Args:
            source_area (str): Identifier of source brain area supplying readout.
            unique_id (str | None, optional): Unique ID string for output connection. Defaults to None (uses 'output').
            trainable (bool, optional): Whether output readout weights are trainable. Defaults to False.
            std (float, optional): Standard deviation of initialized weights. Defaults to 0.0.
            scale (float, optional): Scaling multiplier for initialized weights. Defaults to 1.0.
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
            trainable=trainable)

    def add_custom_connection(
            self,
            connection_name,
            source,
            target,
            initializer,
            trainable=True,
            **initializer_args):
        """
        Add a custom connection to the network architecture using a custom user-defined initializer function.

        Args:
            connection_name (str): Identifier string for custom connection.
            source (str): Source area ID or external source name.
            target (str): Target area ID or external target name.
            initializer (Callable): Custom initializer function returning `(weights, mask)` tensors. The
                function receives a dictionary with network parameters (keys: 'connection', 'source',
                'target', 'general_params', 'model_params') as the first argument, and any optional
                user-specified arguments. The function should return tensors `(weights, mask)` both with
                the shape `(target.size, source.size)`.
            trainable (bool, optional): Whether weights are trainable during optimization. Defaults to True.
            **initializer_args: Additional keyword arguments passed to custom initializer function.
        """
        dales_law_constraint = True if source in self.areas.keys() else False

        connection = Connection(
            connection_name,
            source,
            target,
            trainable,
            dales_law_constraint)

        # If source or target are initialized brain areas, get their BrainArea object
        if source in self.areas.keys():
            source = self.areas[source]
        if target in self.areas.keys():
            target = self.areas[target]

        init_dict = {
            'connection': connection,
            'source': source,
            'target': target,
            'general_params': self.params['general'],
            'model_params': self.params['model']}

        weights, mask = initializer(init_dict, **initializer_args)

        connection.set_sizes(weights.shape[1], weights.shape[0])
        connection.set_weights_and_mask(weights, mask)

        self.connections[connection.get_name()] = connection

    def finalize(self):
        """
        Finalize network geometry after adding all areas and connections, setting state vector slice indices
        and extending adaptation buffers.
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
        Constrain all  connection weight matrices to enforce structural masks and Dale's law sign rules.
        """
        for connection in self.connections.values():
            connection.constrain()

    def run(self, ext_input=None, input_window=None, adjoint=False, stochastic=False, reset_state=True, device="cpu"):
        """
        Delegate numerical integration of network dynamics to the simulation engine across specified time windows.

        Args:
            ext_input (dict | torch.Tensor | None, optional): External drive input specification. Defaults to None.
                If None, the network will run at resting state.
            input_window (dict | tuple | None, optional): Active time window intervals. Defaults to None.
                If None, `input_window` from TOML configuration file will be used.
            adjoint (bool, optional): Whether to use adjoint solver variants. Defaults to False.
            stochastic (bool, optional): Whether to simulate stochastic dynamics with noise. Defaults to False.
            reset_state (bool, optional): Whether to reset initial state to resting state. If False, will use the
                last state of the previous simulation as the current initial state. Defaults to True.
            device (torch.device | str, optional): Compute device for execution. Defaults to "cpu".

        Returns:
            torch.Tensor: Simulation output state trajectory tensor of shape
                `(time_steps, batch_size, 2 * total_populations)`. The last dimension is split into
                membrane potential `[:, :, :N]` and adaptation `[:, :, N:]`
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
        Compute population firing rates from raw state trajectory tensors and filter by sample, area, column,
        or laminar layer.

        Args:
            raw_state (torch.Tensor): Simulation state output tensor.
            sample (int | list[int] | None, optional): Sample indices filter. Defaults to None.
            area (str | None, optional): Target area filter. Defaults to None.
            column (int | list[int] | None, optional): Column indices filter. Defaults to None.
            population (int | str | list | None, optional): Population layer filter. Defaults to None.
            return_as_np_array (bool, optional): Whether to return array as NumPy. Defaults to True.
            return_as_dict (bool, optional): Whether to return structured area dictionary. Defaults to False.

        Returns:
            np.ndarray | torch.Tensor | tuple: Processed firing rates tensor or dictionary tuple.
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
        Compute task output readouts from raw simulation state trajectories using continuous or classification
        time averaging.

        Args:
            raw_output (torch.Tensor): Simulation output state trajectory tensor.
            mode (str): Evaluation mode (`'trajectory'` or `'classification'`).
            sum_per_col (bool, optional): Whether to sum rates across all 8 column populations. Defaults to True.

        Returns:
            torch.Tensor | dict[str, torch.Tensor]: Processed readout tensor or dictionary of readout tensors.
        """
        return self.readout.read_out(
            raw_output=raw_output,
            mode=mode,
            sum_per_col=sum_per_col)

    def save(self, path):
        """
        Serialize and save current network architecture, model parameters, and trained weights to a checkpoint file.

        Args:
            path (str | Path): Output file path for checkpoint saving.
        """
        checkpoint = self.archive.create_checkpoint()
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, model_config_path=None, general_config_path=None):
        """
        Load a saved network checkpoint from file and reinstate its architecture, weights, and parameters.

        Args:
            path (str | Path): File path of saved checkpoint.
            model_config_path (str | Path | None, optional): Optional override config path for model parameters.
                Defaults to None.
            general_config_path (str | Path | None, optional): Optional override config path for general parameters.
                Defaults to None.

        Returns:
            BrainNetwork: Fully reinstated and finalized network instance.
        """
        checkpoint = torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=False)

        return NetworkArchiver.restore_checkpoint(
            cls,
            checkpoint,
            model_config_path,
            general_config_path)