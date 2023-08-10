from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from scipy.spatial.transform import Rotation as R


import numpy as np
import pandas as pd

try:
    # If this file is imported via blender, we can't use `scipy`
    import bpy
except ModuleNotFoundError:
    from scipy.spatial.transform import Rotation


@dataclass
class CameraData:
    """
    Our camera coordinate system uses right-down-forward (RDF) convention similarly to COLMAP.

    Right-handed coordinate system is adopted throughout this project, and vectors are represented as columns.
    So, the transformations need to be applied from the left-hand side, e.g., `t_vector = Matrix @ vector`

    Extrinsics represent the transformation from camera-space to world-space,
    i.e., `world_space = Rotation @ camera_space + Translation`.
    * The magnitude of `rotation_axisangle` defines the rotation angle in radians.
    * `translation` is typically stored in meters [m].
    """

    name: str
    width: int
    height: int

    # Extrinsics
    rotation_axisangle: np.array
    translation: np.array

    # Intrinsics
    focal_length: np.array
    principal_point: np.array

    # Optional distortion coefficients
    k1: float = 0
    k2: float = 0
    k3: float = 0

    @property
    def fx_pixel(self):
        return self.width * self.focal_length[0]

    @property
    def fy_pixel(self):
        return self.height * self.focal_length[1]

    @property
    def cx_pixel(self):
        return self.width * self.principal_point[0]

    @property
    def cy_pixel(self):
        return self.height * self.principal_point[1]

    def intrinsic_matrix(self):
        return np.array(
            [
                [self.fx_pixel, 0, self.cx_pixel],
                [0, self.fy_pixel, self.cy_pixel],
                [0, 0, 1],
            ]
        )

    def rotation_matrix_cam2world(self) -> np.array:
        """Set up the camera to world rotation matrix from the axis-angle representation.

        Returns:
            np.array (3 x 3): Rotation matrix going from camera to world space.
        """
        return Rotation.from_rotvec(self.rotation_axisangle).as_matrix()

    def extrinsic_matrix_cam2world(self) -> np.array:
        """Set up the camera to world transformation matrix to be applied on homogeneous coordinates.

        Returns:
            np.array (4 x 4): Transformation matrix going from camera to world space.
        """
        tfm_cam2world = np.eye(4)
        tfm_cam2world[:3, :3] = self.rotation_matrix_cam2world()
        tfm_cam2world[:3, 3] = self.translation

        return tfm_cam2world

    def projection_matrix_world2pixel(self):
        """Set up the world to pixel transformation matrix to project homogeneous coordinates onto image plane.

        Returns:
            np.array (4 x 4): Transformation matrix going from world to pixel space (division by Z-coordinate must be applied as the last step)
        """
        tfm_world2pixel = np.eye(4)
        tfm_world2pixel[:3] = self.intrinsic_matrix() @ np.linalg.inv(self.extrinsic_matrix_cam2world())[:3]

        return tfm_world2pixel

    def get_downscaled_camera(self, downscale_factor: int) -> CameraData:
        """Get a new `CameraData` object with the same parameters but downscaled by `downscale_factor` along each axis.
        This corresponds to our pre-processing of downscaled versions of the dataset.

        Args:
            scale (int): Downscale factor.

        Returns:
            CameraData: New `CameraData` object with downscaled parameters.
        """
        return CameraData(
            name=self.name,
            width=self.width // downscale_factor,
            height=self.height // downscale_factor,
            rotation_axisangle=self.rotation_axisangle,
            translation=self.translation,
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
        )


def write_calibration_csv(cameras: List[CameraData], output_csv_path: Path) -> None:
    """Write camera intrinsics and extrinsics to a calibration CSV file.

    Args:
        cameras (List[CameraData]): List of `CameraData` objects describing camera parameters.
        output_csv_path (Path): Path to the output CSV file.
    """
    csv_field_names = ["name", "w", "h", "rx", "ry", "rz", "tx", "ty", "tz", "fx", "fy", "px", "py"]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_field_names)
        writer.writeheader()

        for camera in cameras:
            csv_row = {}
            csv_row["name"] = camera.name
            csv_row["w"] = camera.width
            csv_row["h"] = camera.height
            csv_row["rx"] = camera.rotation_axisangle[0]
            csv_row["ry"] = camera.rotation_axisangle[1]
            csv_row["rz"] = camera.rotation_axisangle[2]
            csv_row["tx"] = camera.translation[0]
            csv_row["ty"] = camera.translation[1]
            csv_row["tz"] = camera.translation[2]
            csv_row["fx"] = camera.focal_length[0]
            csv_row["fy"] = camera.focal_length[1]
            csv_row["px"] = camera.principal_point[0]
            csv_row["py"] = camera.principal_point[1]

            assert len(csv_row) == len(csv_field_names)
            writer.writerow(csv_row)


def read_calibration_csv(input_csv_path: Path) -> List[CameraData]:
    """Read camera intrinsics and extrinsics from a calibration CSV file.

    Args:
        input_csv_path (Path): Path to a CSV file that contains camera calibration data.

    Returns:
        List[CameraData]: A list of `CameraData` objects that describe multiple camera intrinsics and extrinsics.
    """
    cameras = []
    with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            camera = CameraData(
                name=row["name"],
                width=int(row["w"]),
                height=int(row["h"]),
                rotation_axisangle=np.array([float(row["rx"]), float(row["ry"]), float(row["rz"])]),
                translation=np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])]),
                focal_length=np.array([float(row["fx"]), float(row["fy"])]),
                principal_point=np.array([float(row["px"]), float(row["py"])]),
            )
            cameras.append(camera)
    return cameras

def rotation_matrix_from_axis_angle(axis_angle):
    """Convert an axis-angle rotation into a rotation matrix."""
    angle = np.linalg.norm(axis_angle)
    axis = axis_angle / angle
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def axis_angle_from_rotation_matrix(rotation_matrix):
    """Convert a rotation matrix into an axis-angle representation."""
    angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (2 * np.sin(angle))
    y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (2 * np.sin(angle))
    z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (2 * np.sin(angle))
    return np.array([x, y, z]) * angle


def look_at_rotation_matrix(look_at_vector):
    # Normalize the look_at_vector
    z_axis = -look_at_vector / np.linalg.norm(look_at_vector)
    
    # Assuming the up-vector is (0, 1, 0)
    x_axis = np.cross(np.array([0, 1, 0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    # Compute the orthogonal up-vector
    y_axis = np.cross(z_axis, x_axis)

    # Construct the rotation matrix
    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=-1)
    
    return rotation_matrix

def rotation_matrix(angle: float, axis: str) -> np.array:
    """
    Creates a 3x3 rotation matrix for a rotation about the specified axis.

    Args:
        angle (float): The rotation angle in radians.
        axis (str): The rotation axis ('x', 'y', or 'z').

    Returns:
        np.array: The rotation matrix.
    """
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])

def rotation_matrix_to_axisangle(rotation: np.array) -> np.array:
    """
    Converts a 3x3 rotation matrix to an axis-angle representation.

    Args:
        rotation (np.array): The rotation matrix.

    Returns:
        np.array: The rotation axis-angle.
    """
    # This is a simplified conversion that only works for rotations about the y-axis
    angle = np.arccos(rotation[0, 0])
    return np.array([0, angle, 0])

def look_at(camera_position, look_at_point, up=np.array([0, 1, 0])):
    """Generate a rotation matrix for a camera that looks at a point.

    Args:
        camera_position (np.array): The 3D position of the camera.
        look_at_point (np.array): The 3D point that the camera is looking at.
        up (np.array, optional): The up direction. Defaults to np.array([0, 1, 0]).

    Returns:
        np.array: A 3x3 rotation matrix.
    """
    forward = (look_at_point - camera_position)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    actual_up = np.cross(forward, right)
    return np.stack([right, actual_up, forward], axis=-1)

def compute_object_center(input_csv_path: Path) -> np.array:
  """Compute the object center based on the average of camera positions.

    Args:
        input_csv_path (Path): Path to a CSV file that contains camera calibration data.

    Returns:
        np.array: A 3D point representing the estimated object center.
    """
  calibration_data = pd.read_csv(input_csv_path)
  translations = calibration_data[['tx', 'ty', 'tz']].values
  center = translations.mean(axis=0)
  return center

def read_calibration_orbited_csv(input_csv_path: Path, actual_cameras_path: Path, num_samples: int, 
                                ) -> List[CameraData]:
    """Read camera intrinsics and extrinsics from a calibration CSV file.

    Args:
        input_csv_path (Path): Path to a CSV file that contains camera calibration data.
        actual_cameras_path (Path): Path to a CSV file that contains real camera calibration data to calculate the center of the object.
        num_samples (int): Number of sampled cameras in the orbit

    Returns:
        List[CameraData]: A list of `CameraData` objects that describe multiple camera intrinsics and extrinsics and orbited ones. 
    """
    object_center = compute_object_center(actual_cameras_path)
    radius = compute_average_radius(actual_cameras_path, object_center)

    if object_center is None:
        object_center = np.array([0, 0, 0])
    if up is None:
        up = np.array([0, -1, 0])

    csv_cameras = read_calibration_csv(input_csv_path)
    cameras = []
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    for curr_camera in csv_cameras:
      cameras.append(curr_camera)
      for i, angle in enumerate(angles):
        camera_position = object_center + radius * np.array([np.cos(angle), 0, np.sin(angle)])       
        rotation_matrix = look_at(camera_position, object_center, up)      

        new_camera = CameraData(
            name=f"{curr_camera.name}_orbit_{i}",
            width = curr_camera.width,
            height = curr_camera.height,
            rotation_axisangle = Rotation.from_matrix(rotation_matrix).as_rotvec(),
            translation = camera_position,
            focal_length = curr_camera.focal_length.copy(),
            principal_point = curr_camera.principal_point.copy(),
        )
        cameras.append(new_camera)
    return cameras

def compute_average_radius(calibration_data_path: Path, object_center: np.array) -> float:
    """Compute the average radius based on the distances of camera positions to the object center.

    Args:
        calibration_data (pd.DataFrame): DataFrame containing camera calibration data.
        object_center (np.array): A 3D point representing the estimated object center.

    Returns:
        float: The average radius.
    """
    calibration_data = pd.read_csv(calibration_data_path)
    camera_positions = calibration_data[['tx', 'ty', 'tz']].values
    distances = np.linalg.norm(camera_positions - object_center, axis=1)
    average_radius = distances.mean()
    return average_radius

def read_calibration_uniformed_csv(input_csv_path: Path, actual_cameras_path: Path, num_samples: int, radius: Optional[float] = None,
                                ) -> List[CameraData]:
    """Read camera intrinsics and extrinsics from a calibration CSV file.

    Args:
        input_csv_path (Path): Path to a CSV file that contains camera calibration data.
        actual_cameras_path (Path): Path to a CSV file that contains real camera calibration data to calculate the center of the object.
        num_samples (int): Number of sampled cameras in the orbit
        radius (float): Radius of the sphere on which sampling will be done.
    Returns:
        List[CameraData]: A list of `CameraData` objects that describe multiple camera intrinsics and extrinsics and uniformed ones. 
    """
    object_center = compute_object_center(actual_cameras_path)
    if object_center is None:
        object_center = np.array([0, 0, 0])

    if radius is None:
      radius = compute_average_radius(actual_cameras_path, object_center)

    #Global up direction in RDF convention
    up = np.array([0, -1, 0])

    csv_cameras = read_calibration_csv(input_csv_path)
    cameras = []
    phi_values = np.linspace(0, np.pi, int(np.sqrt(num_samples)))  # Polar angle
    theta_values = np.linspace(0, 2 * np.pi, int(num_samples / len(phi_values)))  # Azimuthal angle


    commmon_width=csv_cameras[0].width
    common_height=csv_cameras[0].height
    common_focal_length=csv_cameras[0].focal_length.copy()
    common_principal_point=csv_cameras[0].principal_point.copy()

    for phi in phi_values:
        for theta in theta_values:
            # Convert spherical coordinates to Cartesian coordinates
            x = object_center[0] + radius * np.sin(phi) * np.cos(theta)
            y = object_center[1] + radius * np.sin(phi) * np.sin(theta)
            z = object_center[2] + radius * np.cos(phi)
            camera_position = np.array([x, y, z])

            # Compute rotation matrix using the look_at function
            rotation_matrix = look_at(camera_position, object_center, up) 

            
            # Create CameraData object for the sampled camera
            camera = CameraData(
                name=f"CamSphere_{len(cameras) + 1}",
                width= commmon_width,
                height= common_height,
                rotation_axisangle=Rotation.from_matrix(rotation_matrix).as_rotvec(),
                translation=camera_position,
                focal_length=common_focal_length,
                principal_point=common_principal_point,
            )
            cameras.append(camera)

    return cameras


def read_calibration_list() -> List[CameraData]:
    """Read camera intrinsics and extrinsics from a list.

    Returns:
        List[CameraData]: A list of `CameraData` objects that describe multiple camera intrinsics and extrinsics.
    """
    hardcoded_cams = [{'name': 'Cam011', 'w': '870', 'h': '1022', 'rx': '3.621405502961905', 'ry': '-0.007217732617988849', 'rz': '-0.07128564910350631', 'tx': '0.16223773393442864', 'ty': '0.29387954499352065', 'tz': '2.4078490858217676', 'fx': '1.6709186729726866', 'fy': '1.3045395343074269', 'px': '0.5013477333233055', 'py': '0.49994617379852013'},
                      {'name': 'Cam012', 'w': '888', 'h': '940', 'rx': '-5.216654624994702', 'ry': '0.024553674256727765', 'rz': '0.14314215740134242', 'tx': '-0.4000057231332983', 'ty': '0.7181807246046349', 'tz': '2.3934111749108085', 'fx': '1.6503508352032394', 'fy': '1.5908957826515941', 'px': '0.5021861706874694', 'py': '0.49576949396498815'},
                      {'name': 'Cam013', 'w': '925', 'h': '735', 'rx': '4.138448841495705', 'ry': '-0.029206575469255775', 'rz': '0.060954254051468454', 'tx': '0.3175898870076602', 'ty': '0.8695226876502581', 'tz': '2.145382567849009', 'fx': '1.9201470768051816', 'fy': '1.8656596916928985', 'px': '0.5016769982487265', 'py': '0.5076824771233476'},
                      {'name': 'Cam014', 'w': '746', 'h': '795', 'rx': '-3.3727680040569594', 'ry': '-0.04156916932833471', 'rz': '-0.16900689421625423', 'tx': '-0.05907628083304489', 'ty': '1.9767502388899016', 'tz': '2.120421932372801', 'fx': '1.586205078730376', 'fy': '3.067727554141795', 'px': '0.50586328294691', 'py': '0.5014064575146523'},
                      {'name': 'Cam015', 'w': '977', 'h': '687', 'rx': '3.659297612280421', 'ry': '-0.011310346065601475', 'rz': '-0.18524507224703002', 'tx': '0.28158967119345596', 'ty': '1.1294590443413015', 'tz': '2.24610713760478', 'fx': '1.7524566113853368', 'fy': '2.285500282472767', 'px': '0.4981798202527454', 'py': '0.5073735263069437'},
                      {'name': 'Cam016', 'w': '830', 'h': '1108', 'rx': '-1.678225496376341', 'ry': '-0.036204838316383464', 'rz': '-0.24480425200608624', 'tx': '-0.526598709914843', 'ty': '0.1984851014364546', 'tz': '2.1290209342566238', 'fx': '1.6049332402067225', 'fy': '2.724111317058074', 'px': '0.5123794911598912', 'py': '0.5004470691511821'},
                      {'name': 'Cam017', 'w': '1088', 'h': '718', 'rx': '0.011396061025858245', 'ry': '0.017168331302112948', 'rz': '-0.35961459979910465', 'tx': '0.14642344971060656', 'ty': '1.5803346890159848', 'tz': '2.240771414898348', 'fx': '1.5822656890579718', 'fy': '2.50236223626876', 'px': '0.49928501224468363', 'py': '0.49714377889858424'},
                      {'name': 'Cam018', 'w': '863', 'h': '800', 'rx': '0.7256933947522196', 'ry': '-0.04540087837030354', 'rz': '-0.004707886360442195', 'tx': '-0.4177828518797869', 'ty': '-0.5670545459300012', 'tz': '2.083118689137067', 'fx': '1.7394508636573656', 'fy': '1.6959323967944333', 'px': '0.5016526004662346', 'py': '0.4953854697719054'},
                      {'name': 'Cam019', 'w': '935', 'h': '611', 'rx': '-1.325502293526586', 'ry': '0.004228124455538266', 'rz': '-0.3714890841407946', 'tx': '0.5161049229760789', 'ty': '0.6083999725558229', 'tz': '2.090698051644113', 'fx': '1.8667188052824653', 'fy': '1.785794997552897', 'px': '0.5025526190662141', 'py': '0.49698805925532336'}]
    cameras = []
    for camera_info in hardcoded_cams:
        camera = CameraData(
            name=camera_info["name"],
            width=int(camera_info["w"]),
            height=int(camera_info["h"]),
            rotation_axisangle=np.array([float(camera_info["rx"]), float(camera_info["ry"]), float(camera_info["rz"])]),
            translation=np.array([float(camera_info["tx"]), float(camera_info["ty"]), float(camera_info["tz"])]),
            focal_length=np.array([float(camera_info["fx"]), float(camera_info["fy"])]),
            principal_point=np.array([float(camera_info["px"]), float(camera_info["py"])]),
        )
        cameras.append(camera)
    return cameras


