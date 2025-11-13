import numpy as np
from scipy.spatial.transform import Rotation


def get_carla_object_current_yaw(tracks):
    quaternion_names = ["carla_objects_pose_orientation_" + axis for axis in ["x", "y", "z", "w"]]
    rot = Rotation.from_quat(tracks[quaternion_names].iloc[-1].to_numpy())
    return rot.as_euler("xyz", degrees=False)[-1] - (np.pi / 2.0)


def get_carla_object_yaw(tracks, index=None):
    quaternion_names = ["carla_objects_pose_orientation_" + axis for axis in ["x", "y", "z", "w"]]
    if index is None:
        rot = Rotation.from_quat(tracks.loc[:, quaternion_names].to_numpy())
        return rot.as_euler("xyz", degrees=False)[:, -1] - (np.pi / 2.0)
    else:
        rot = Rotation.from_quat(tracks.loc[index, quaternion_names].to_numpy())
        return rot.as_euler("xyz", degrees=False)[-1] - (np.pi / 2.0)


def get_carla_tuple_yaw(tracks):
    quaternion_names = ["carla_objects_pose_orientation_" + axis for axis in ["x", "y", "z", "w"]]
    rot = []
    for quaternion_name in quaternion_names:
        rot.append(getattr(tracks, quaternion_name))
    rot = Rotation.from_quat(np.array(rot))
    return rot.as_euler("xyz", degrees=False)[-1]
