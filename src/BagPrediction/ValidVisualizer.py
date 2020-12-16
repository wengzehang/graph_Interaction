"""Visualization for simulated data

"""

import open3d
from datetime import datetime
import numpy as np
from typing import List, Union, Tuple

import SimulatedData

# from DataVisualizer import Frame
import DataVisualizer

import GraphRepresentation
import GraphNetworkModules
import DataGenerator
from graph_nets import utils_tf
import copy
import sonnet as snt
import tensorflow as tf
import os

# TODO: This should be configurable
kpset = [759, 545, 386, 1071, 429, 943, 820, 1013, 1124, 1212, 1269, 674, 685, 1236]

class KeypointDataVisualizer:
    # The video_id/visid is the scenario index, i.e. a single task execution
    def __init__(self, data: SimulatedData, scenario_index: int,
                 keypoint_indices: List[int], keypoint_edges: List[Tuple[int, int]]):
        self.data = data
        self.scenario_index = scenario_index
        self.frame_index = 0
        self.keypoint_index = 0
        shape = data.dataset[SimulatedData.MESH_KEY].shape
        self.num_scenarios = shape[0]
        self.num_frames = shape[1]
        self.num_mesh_points = shape[2]
        self.running = True
        self.show_cloth_mesh = True

        self.keypoint_indices = keypoint_indices
        self.keypoint_edges = keypoint_edges
        self.keypoint_edges_indices = [(keypoint_indices.index(f), keypoint_indices.index(t))
                                       for (f, t) in keypoint_edges]

        self.dataset_cloth = self.data.dataset[SimulatedData.MESH_KEY][:]

        self.frames = self.load_frames()


    def load_frames(self) -> List[DataVisualizer.Frame]:
        return [self.create_frame(i) for i in range(self.num_frames)]

    def create_frame(self, frame_index: int):

        dataset = self.data.dataset

        num_rigid = dataset[SimulatedData.RIGID_NUM_KEY][self.scenario_index]
        cloth_id = dataset[SimulatedData.CLOTH_ID_KEY][self.scenario_index]
        seq_rigid = dataset[SimulatedData.RIGID_KEY][self.scenario_index, frame_index, :num_rigid, :]  # (numrigid, 4), xyzr

        mesh_tx_list = []
        for obj_i in range(num_rigid):
            # get the origin of each rigid object
            xyz = seq_rigid[obj_i][0:3]
            # get the radius of each rigid object
            r = seq_rigid[obj_i][3]
            # create the sphere in open3d
            mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=r)
            # translate the sphere object according to the origin position
            mesh_tx = mesh_sphere.translate(xyz)
            # mesh_tx.compute_vertex_normals()
            # mesh_tx.paint_uniform_color([0.1, 0.1, 0.7])
            mesh_tx_list.append(mesh_tx)

        seq = self.dataset_cloth[self.scenario_index, frame_index, :]  # (numpoint, 3)
        # g1 = copy.copy(seq[:, 0])
        # g2 = copy.copy(seq[:, 2])
        # seq[:,2] = g1
        # seq[:,0] = g2
        conn = self.data.topodict[cloth_id]
        cloth_mesh = open3d.geometry.TriangleMesh()
        cloth_mesh.vertices = open3d.utility.Vector3dVector(seq)
        cloth_mesh.triangles = open3d.utility.Vector3iVector(conn)
        cloth_mesh.vertex_colors = open3d.utility.Vector3dVector(np.full((seq.shape[0], 3), 0.5))

        if self.show_cloth_mesh:
            mesh_tx_list.append(cloth_mesh)

        #######################
        total_points = seq
        cloth_pcd = open3d.geometry.PointCloud()
        color_point = np.full(total_points.shape, np.array([0.0, 0.0, 0.0]))
        for i in self.keypoint_indices:
            color_point[i] = np.array([0.8, 0.0, 0.0])
        color_point[self.keypoint_index] = np.array([1.0, 0.0, 0.0])

        if self.show_cloth_mesh:
            cloth_pcd.points = open3d.utility.Vector3dVector(total_points)
            cloth_pcd.colors = open3d.utility.Vector3dVector(color_point)
        else:
            cloth_pcd.points = open3d.utility.Vector3dVector(total_points[self.keypoint_indices])
            cloth_pcd.colors = open3d.utility.Vector3dVector(color_point[self.keypoint_indices])
        mesh_tx_list.append(cloth_pcd)

        ##########################
        # effector
        seq_effector = dataset[SimulatedData.EFFECTOR_KEY][self.scenario_index, frame_index, :][0]
        effector_xyz = seq_effector[0:3]
        effector_r = seq_effector[3]
        mesh_sphere_effector = open3d.geometry.TriangleMesh.create_sphere(radius=effector_r)
        # translate the sphere object according to the origin position
        mesh_tx_effector = mesh_sphere_effector.translate(effector_xyz)
        # mesh_tx_effector = mesh_sphere_effector
        mesh_tx_effector.paint_uniform_color([0.1, 0.1, 0.7])
        # mesh_tx.compute_vertex_normals()
        # mesh_tx.paint_uniform_color([0.1, 0.1, 0.7])
        mesh_tx_list.append(mesh_tx_effector)

        total_points_e = seq_effector[0:3].reshape(-1, 3)
        pcd_e = open3d.geometry.PointCloud()
        pcd_e.points = open3d.utility.Vector3dVector(total_points_e)
        color_point_e = np.zeros(total_points_e.shape)
        pcd_e.colors = open3d.utility.Vector3dVector(color_point_e)
        mesh_tx_list.append(pcd_e)

        # Add line set between keypoints
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(seq[self.keypoint_indices])
        line_set.lines = open3d.utility.Vector2iVector(np.array(self.keypoint_edges_indices))
        line_set.colors = open3d.utility.Vector3dVector(np.full((len(self.keypoint_edges_indices), 3),
                                                                [1.0, 0.0, 1.0]))
        mesh_tx_list.append(line_set)

        return DataVisualizer.Frame(mesh_tx_list, cloth_mesh, cloth_pcd, color_point)

    def on_quit(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.running = False

    def on_prev_scenario(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.scenario_index = self.scenario_index - 1
        if self.scenario_index < 0:
            self.scenario_index = self.num_scenarios - 1
        self.frame_index = 0

    def on_next_scenario(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.scenario_index = self.scenario_index + 1
        if self.scenario_index >= self.num_scenarios:
            self.scenario_index = 0
        self.frame_index = 0

    def on_prev_frame(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.frame_index = self.frame_index - 1
        if self.frame_index < 0:
            self.frame_index = len(self.frames) - 1

    def on_next_frame(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.frame_index = self.frame_index + 1
        if self.frame_index >= len(self.frames):
            self.frame_index = 0

    def on_prev_keypoint(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.keypoint_index = self.keypoint_index - 1
        if self.keypoint_index < 0:
            self.keypoint_index = self.num_mesh_points - 1

    def on_next_keypoint(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.keypoint_index = self.keypoint_index + 1
        if self.keypoint_index >= self.num_mesh_points:
            self.keypoint_index = 0

    def on_toggle_cloth_mesh(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.show_cloth_mesh = not self.show_cloth_mesh

    def register_callbacks(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        GLFW_KEY_ESCAPE = 256
        GLFW_KEY_RIGHT = 262
        GLFW_KEY_LEFT = 263
        GLFW_KEY_DOWN = 264
        GLFW_KEY_UP = 265
        GLFW_KEY_A = 65
        GLFW_KEY_S = 83
        GLFW_KEY_SPACE = 32

        o3d_vis.register_key_callback(GLFW_KEY_ESCAPE, self.on_quit)
        o3d_vis.register_key_callback(GLFW_KEY_LEFT, self.on_prev_frame)
        o3d_vis.register_key_callback(GLFW_KEY_RIGHT, self.on_next_frame)
        o3d_vis.register_key_callback(GLFW_KEY_DOWN, self.on_prev_scenario)
        o3d_vis.register_key_callback(GLFW_KEY_UP, self.on_next_scenario)
        o3d_vis.register_key_callback(GLFW_KEY_A, self.on_prev_keypoint)
        o3d_vis.register_key_callback(GLFW_KEY_S, self.on_next_keypoint)
        o3d_vis.register_key_callback(GLFW_KEY_SPACE, self.on_toggle_cloth_mesh)

    def run(self):
        o3d_vis = open3d.visualization.VisualizerWithKeyCallback()
        o3d_vis.create_window()
        o3d_vis.get_render_option().point_size = 10
        o3d_vis.get_render_option().line_width = 5
        o3d_vis.get_render_option().show_coordinate_frame = True

        self.register_callbacks(o3d_vis)

        old_scenario_index = -1
        old_frame_index = -1
        old_keypoint_index = -1
        old_show_cloth_mesh = self.show_cloth_mesh
        self.running = True
        while self.running:
            # running, scenario_index and frame_index are modified by the keypress events
            scenario_changed = self.scenario_index != old_scenario_index
            show_cloth_mesh_changed = self.show_cloth_mesh != old_show_cloth_mesh

            old_scenario_index = self.scenario_index
            if scenario_changed or show_cloth_mesh_changed:
                self.frames = self.load_frames()

            frame_changed = self.frame_index != old_frame_index
            keypoint_changed = self.keypoint_index != old_keypoint_index

            frame = self.frames[self.frame_index]
            if keypoint_changed:
                was_keypoint = old_keypoint_index in self.keypoint_indices
                frame.update_keypoint(old_keypoint_index, self.keypoint_index, was_keypoint, o3d_vis)
                old_keypoint_index = self.keypoint_index

            if scenario_changed or frame_changed or keypoint_changed or show_cloth_mesh_changed:
                # Only reset the bounding box if we load a new scene
                print("Rendering frame: ", self.scenario_index, ":", self.frame_index)
                frame.render(o3d_vis, reset_bounding_box=scenario_changed)
                old_frame_index = self.frame_index
                old_show_cloth_mesh = self.show_cloth_mesh

            o3d_vis.poll_events()
            o3d_vis.update_renderer()

        o3d_vis.destroy_window()

def compute_output(module, inputs_tr):
    outputs_tr = module(inputs_tr)
    return outputs_tr

def create_loss(target, outputs):
    losses = [
        tf.compat.v1.losses.mean_squared_error(target.nodes, output.nodes) +
        tf.compat.v1.losses.mean_squared_error(target.edges, output.edges)
        for output in outputs
    ]
    return tf.stack(losses)

def compute_output_and_loss(module, inputs_tr, targets_tr):
    outputs_tr = module(inputs_tr)
    loss_tr = create_loss(targets_tr, outputs_tr)
    loss_tr = tf.math.reduce_sum(loss_tr) / module.num_processing_steps
    return outputs_tr, loss_tr

# Create the graph network.
def make_mlp(layers):
    return snt.Sequential([
        snt.nets.MLP(layers, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])

def snt_mlp(layers):
    return lambda: make_mlp(layers)

module = GraphNetworkModules.EncodeProcessDecode(
    make_encoder_edge_model=snt_mlp([64, 64]),
    make_encoder_node_model=snt_mlp([64, 64]),
    make_encoder_global_model=snt_mlp([64]),
    make_core_edge_model=snt_mlp([64, 64]),
    make_core_node_model=snt_mlp([64, 64]),
    make_core_global_model=snt_mlp([64]),
    num_processing_steps=5,
    edge_output_size=3,
    node_output_size=3,
    global_output_size=1,
)

if __name__ == '__main__':
    # valid_path_to_topodict = 'h5data/topo_valid.pkl'
    # valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1.h5'
    valid_path_to_topodict = 'h5data/topo_train.pkl'
    valid_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1.h5'

    representation = GraphRepresentation.GraphRepresentation(SimulatedData.keypoint_indices,
                                                             SimulatedData.keypoint_edges)

    data = SimulatedData.SimulatedData.load(valid_path_to_topodict, valid_path_to_dataset)

    newdata = copy.copy(data)
    # feed the whole validation dataset into the graph network and get the output graph state
    startframe = 0
    # load the trained graph module
    graphmodule = module
    num_processing_steps = 5

    # Checkpoint stuff
    model_path = "./models/test-1"
    checkpoint_root = model_path + "/checkpoints"
    checkpoint_name = "checkpoint-1"
    checkpoint_save_prefix = os.path.join(checkpoint_root, checkpoint_name)
    checkpoint = tf.train.Checkpoint(module=module)
    latest = tf.train.latest_checkpoint(checkpoint_root)
    checkpoint.restore(latest)

    #
    print("Loading latest checkpoint: ", latest)

    scenario_index = 0
    keypoint_indices = SimulatedData.keypoint_indices
    keypoint_edges = SimulatedData.keypoint_edges
    SimulatedData.validate_keypoint_graph(keypoint_indices, keypoint_edges)

    # data_vis = DataVisualizer(data, scenario_index, keypoint_indices, keypoint_edges)
    data_vis = KeypointDataVisualizer(newdata, scenario_index, keypoint_indices, keypoint_edges)

    prev_input_graph_tuples = None
    # for i_scenario in range(newdata.num_scenarios):
    for i_scenario in range(10):
        print("done with {} scene.".format(i_scenario))
        for i_frame in range(newdata.num_frames):
            if i_frame == 0:
                scenario = newdata.scenario(i_scenario)
                prev_frame = scenario.frame(i_frame)
                prev_graph_dict = representation.to_graph_dict(prev_frame)
                prev_input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple([prev_graph_dict])
            else:
                current_predict_tuples = compute_output(module=graphmodule, inputs_tr=prev_input_graph_tuples)

                data_vis.dataset_cloth[i_scenario][i_frame][
                        representation.keypoint_indices] = current_predict_tuples[-1].nodes
                prev_input_graph_tuples = current_predict_tuples[-1]


    data_vis.run()
