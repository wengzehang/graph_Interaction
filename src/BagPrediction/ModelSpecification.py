"""
Model specification for unified training and prediction models
"""

from enum import Enum
from typing import List


class NodeFormat(Enum):
    """
    Node attribute format

    Dummy: One float is used as a dummy attribute.
    XYZR: The position (XYZ) and the radius (R) is encoded as node attribute
    HasMovedClasses: Two-class classification result ([1.0 0.0] has not moved, [0.0 1.0] has moved)
    TODO: What about fixed flags?
    """
    Dummy = 0
    XYZR = 1
    HasMovedClasses = 2


class EdgeFormat(Enum):
    """
    Edge attribute format

    Dummy: One float is used as a dummy attribute.
    DiffXYZ: Position difference (XYZ) to the adjacent node.
    """
    Dummy = 0
    DiffXYZ = 1


class GlobalFormat(Enum):
    """
    Global attribute format

    Dummy: One float is used as a dummy attribute.
    NextEndEffectorXYZ: Next position (XYZ) of the end effector (ball).
    NextHandPositionXYZ: Next position (XYZ) of the hand holding part of the bag.
    TODO: Right hand, left hand?
    """
    Dummy = 0
    NextEndEffectorXYZ = 1
    NextHandPositionXYZ = 2


class PositionFrame(Enum):
    """
    Frame used for position attributes.

    Global: The global frame.
    LocalToEndEffector: A frame local to the end effector position (ball).
    """
    Global = 0
    LocalToEndEffector = 1


class GraphAttributeFormat:
    def __init__(self,
                 node_format: NodeFormat = NodeFormat.Dummy,
                 edge_format: EdgeFormat = EdgeFormat.Dummy,
                 global_format: GlobalFormat = GlobalFormat.Dummy,
                 position_frame: PositionFrame = PositionFrame.LocalToEndEffector):
        self.node_format = node_format
        self.edge_format = edge_format
        self.globals = global_format
        self.position_frame = position_frame


class NodeActivationFunction(Enum):
    """
    Node activation function on a graph network.

    Linear: Linear activation function (used for regression)
    Softmax: Softmax activation function (used for classification)
    """
    Linear = 0
    Softmax = 1


class GraphNetStructure:
    def __init__(self,
                 encoder_edge_layers: List[int] = None,
                 encoder_node_layers: List[int] = None,
                 encoder_global_layers: List[int] = None,
                 core_edge_layers: List[int] = None,
                 core_node_layers: List[int] = None,
                 core_global_layers: List[int] = None,
                 num_processing_steps=5,
                 edge_output_size=4,
                 node_output_size=5,
                 global_output_size=4,
                 node_activation_function=NodeActivationFunction.Linear):
        self.encoder_edge_layers = encoder_edge_layers
        self.encoder_node_layers = encoder_node_layers
        self.encoder_global_layers = encoder_global_layers
        self.core_edge_layers = core_edge_layers
        self.core_node_layers = core_node_layers
        self.core_global_layers = core_global_layers
        self.num_processing_steps = num_processing_steps
        self.edge_output_size = edge_output_size
        self.node_output_size = node_output_size
        self.global_output_size = global_output_size
        self.node_activation_function = node_activation_function


class ModelSpecification:
    """
    Specification of a trainable model.

    This specification is sufficient to
    - (re-)create the graph network architecture
    - convert input data (SimulatedData) into input graphs for the network
    - convert ground-truth data (SimulatedData) into target graphs for training the network
    - convert the network output back into frame data (PredictedFrame)
    - ... (Maybe more)
    TODO: Implement training based on this model specification
        - a) Implement conversion of input data (SimulatedData) into the desired input graph format
        - b) Implement conversion of output data (GraphsTuple) into predicted frame (PredictedFrame)
            ==> This is not always possible (Has moved classification)
        - c) Create an EncodeProcessDecode architecture based on spec
        - d) Do training based on spec (Do we need hyper parameters here as well?)

    TODO: Implement evaluation based on this model specification
        - a) Load weights from saved checkpoints
        - b) Implement predict step with the loaded model
        - c)


    The specification consists of the following attributes:
    - input_graph_format: The frame data is converted into this input graph format (for training and prediction)
    - output_graph_format: The network output format (also needs to be generated from ground-truth data)
    - graph_net_structure: The structure of the Encode-Process-Decode architecture (layer sizes and output functions)
    """

    def __init__(self,
                 input_graph_format: GraphAttributeFormat = None,
                 output_graph_format: GraphAttributeFormat = None,
                 graph_net_structure: GraphNetStructure = None
                 ):
        self.input_graph_format = input_graph_format
        self.output_graph_format = output_graph_format
        self.graph_net_structure = graph_net_structure
