from torch.utils import data
import numpy as np
import random 
import torch
from copy import deepcopy

class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        
        # print(self.edge_types)

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            # present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            present_node_dict = scene.present_nodes(np.arange(50, 51), type=self.node_type, **kwargs)#只取第50帧为当前时间步
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    valid = True
                    # valid = False
                    # print("t = ",t)
                    data = [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)
                    (scene, t, node) = data[0]
                    if self.augment:
                        scene = scene.augment()
                        node = scene.get_node_by_id(node.id)
                    data = get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,\
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
                    # if data is None:
                    #     pass  # 跳过不满足条件的轨迹
                    # else:
                    #     valid = True
                    first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name, timestep, velocity = data
                    all_t = torch.cat((x_t[:,:2], y_t),dim=0)
                    if valid:
                        # print("scene_name: ",scene_name)
                        # print("node_id: ", )
                        index += [ (first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name, timestep, velocity, node.id)]
                    else: 
                        pass
        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name, timestep, velocity, node_id) = self.index[i]
        return first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name, timestep, velocity, node_id


def get_node_timestep_data(env, scene, t, node, state, pred_state,
                           edge_types, max_ht, max_ft, hyperparams,
                           scene_graph=None):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """
    # # 仅在 t = 50 时处理，并且确保轨迹长度满足条件
    # if t != 50 or node.history_points_at(t) < 50 + max_ft:
    #     return None  # 跳过不满足条件的轨迹

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])
    # print("t = ",t)
    # print("max_ht = ",max_ht)
    # print("max_ft = ",max_ft)
    x = node.get(timestep_range_x, state[node.type])
    
    velocity = x[:, 2:4]
    y = node.get(timestep_range_y, pred_state[node.type])
    
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)
    # print("node.history_points_at(t) = ", node.history_points_at(t))
    # print("first_history_index", first_history_index)
    x_st_t = deepcopy(x)
    x_st_t = x_st_t - x[-1]
    y_st_t = y
    

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    # print("x shape", x.shape)
    # print("y shape", y.shape)

    x_st_t = torch.tensor(x_st_t, dtype=torch.float)
    y_st_t = torch.tensor(y_st_t, dtype=torch.float)
    # print("scene.name",scene.name)
    return (first_history_index, x_t, y_t, x_st_t, y_st_t, scene.name, t, velocity)
