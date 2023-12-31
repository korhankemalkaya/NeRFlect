import importlib
import sys
import os
import csv
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from simple_parsing import ArgumentGenerationMode, ArgumentParser, NestedMode, field
from actorshq.dataset.camera_data import CameraData

from humanrf.args.model_args import _model_args


@dataclass
class _training_args:
    # name of the set of predefined cameras to use during training.
    camera_preset: str
    # maximum number of steps to be performed for training.
    max_steps: int
    # scaling factor to modulate the loss for mixed-precision training.
    scaler_growth_interval: int
    # can be 'latest', 'best' or path to the '.pth' file.
    checkpoint: str = "latest"
    # initial learning rate.
    lr: float = 1e-2
    # total decay in the learning rate until the end of the training.
    lr_decay: float = 0.5
    # the number of rays to sample for one batch initially.
    rays_initial_batch_size: int = 8192
    # maximum number of samples over all the rays in one batch.
    samples_max_batch_size: int = 768_000
    # weight of the regularization loss via masks.
    bce_loss_weight: float = 1e-3
    # every how many training steps (=iterations) to save a checkpoint.
    save_checkpoint_every_n_steps: int = 2500


@dataclass
class _validation_args:
    # name of the set of predefined cameras to use during validation.
    camera_preset: str
    # determines which (camera, frame) pairs to be used during evaluation.
    # see 'actorshq.evaluation.presets.get_render_sequence()' for details.
    coverage: str = field(default="uniform", choices=["exhaustive", "uniform"])
    # there will be [repeat_cameras] * [# validation cameras] many images for validation.
    repeat_cameras: int = 1
    # determines the frequency of performing validation.
    every_n_steps: int = 2500
    # the number of rays to sample for each batch of validation. validation renders full images, this option is
    # to simply prevent OOM.
    rays_batch_size: int = 8192


@dataclass
class _test_args:
    # can be 'latest', 'best' or path to the '.pth' file.
    checkpoint: str = "best"
    # test video is generated on a trajectory based on these camera indices (0-indexed).
    trajectory_via_keycams: Optional[Tuple[int, ...]] = None
    # if test.trajectory_via_keycams option is used to render a trajectory, the number of cameras to generate for the
    # entire trajectory is determined by this parameter.
    trajectory_num_cameras: int = 200
    # test video is generated on a trajectory based on a calibration.csv.
    trajectory_via_calibration_file: Optional[Path] = None
    # the number of rays to sample for each batch of test. test renders full images, this option is
    # to simply prevent OOM.
    rays_batch_size: int = 16384


@dataclass
class _evaluation_args:
    # name of the set of predefined cameras to use during evaluation.
    camera_preset: str
    # determines which (camera, frame) pairs to be used during evaluation.
    # see 'actorshq.evaluation.presets.get_render_sequence()' for details.
    coverage: str = field(choices=["siggraph_test", "exhaustive", "uniform"])
    # frame numbers to be used during evaluation. if None, dataset.frame_numbers is used for evaluation.
    frame_numbers: Optional[Tuple[int, ...]] = None
    # the number of rays to sample for each batch of validation. validation renders full images, this option is
    # to simply prevent OOM.
    rays_batch_size: int = 16384


@dataclass
class _dataset_args:
    # path to the folder that contains downloaded dataset.
    path: Path
    # actor to use.
    actor: str
    # sequence to use.
    sequence: str
    # downscaling factor of data.
    scale: int
    # whether to crop the center square from each image for training and validation
    crop_center_square: bool
    # whether to use light source annotations to filter light bloom effect.
    filter_light_bloom: bool
    # frame numbers to train & validate & test.
    frame_numbers: Tuple[int, ...]
    # upper limit on the size of the pool of images from which the rays are sampled.
    max_buffer_size: int = 200
    # indicates the number of distinct frames that could exist in the pool from which the rays are sampled.
    max_num_frames_per_batch: int = 8

def _generate_temp_csv(camera_parameters: str):
  # Create directory if it doesn't exist
  directory = "temp_input"
  if not os.path.exists(directory):
      os.makedirs(directory)

  csv_file_path = os.path.join(directory, 'calibration.csv')

  with open(csv_file_path, 'w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=['name', 'w', 'h', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'fx', 'fy', 'px', 'py'])
      writer.writeheader()

      params = camera_parameters.split(',')
      if len(params) != 13:  # Check if there are exactly 13 parameters
        raise ValueError("Invalid number of camera parameters provided!")

      name, w, h, rx, ry, rz, tx, ty, tz, fx, fy, px, py = params
      
      # Type casting the parameters
      w = int(w)
      h = int(h)
      rx = float(rx)
      ry = float(ry)
      rz = float(rz)
      tx = float(tx)
      ty = float(ty)
      tz = float(tz)
      fx = float(fx)
      fy = float(fy)
      px = float(px)
      py = float(py)

      writer.writerow({
          'name': name, 'w': w, 'h': h, 'rx': rx, 'ry': ry, 'rz': rz,
          'tx': tx, 'ty': ty, 'tz': tz, 'fx': fx, 'fy': fy, 'px': px, 'py': py
      })

  return csv_file_path

def _generate_additional_center(object_center_addition: str):
 return np.array([float(x) for x in object_center_addition.split(',')])

@dataclass
class _run_args:
    # perform training if true.
    train: bool
    # perform evaluation if true.
    evaluate: bool
    # perform orbitation if true
    is_orbited: bool
    # perform uniformly sampling if true
    is_uniformed: bool
    # indicates the number of newly generated cameras
    sample_number: int
    # the outputs and training progress will be saved in this local folder.
    workspace: Path
    # model-related parameters
    model: _model_args
    # training-related parameters
    training: _training_args
    # validation-related parameters
    validation: _validation_args
    # evaluation-related parameters
    evaluation: _evaluation_args
    # dataset-related parameters
    dataset: _dataset_args
    # indicates the camera parameters if it is given manually
    camera_parameters: Optional[str] = None
    # indicates the adjustable object center parameters
    object_center_addition: Optional[str] = None
    # indicates the radius of circle or sphere according to sampling method
    radius: Optional[float] = None
    #indicates the specific frame in case a particular one is wanted.
    specific_frame: Optional[int] = None
    # name of the config file (without .py extension) residing under configs/
    config: Optional[str] = None
    # random seed for any source of random numbers.
    random_seed: int = 123
    # can be 'cpu' or 'cuda'.
    device: str = "cuda"
    # test-related parameters
    test: _test_args = _test_args()


def parse_args() -> _run_args:
    cli_args = sys.argv[1:]

    if "--config" in cli_args:
        module_name = cli_args[cli_args.index("--config") + 1]
        sys.argv = sys.argv[0:1] + importlib.import_module(f"humanrf.configs.{module_name}").config + sys.argv[1:]

    parser = ArgumentParser(argument_generation_mode=ArgumentGenerationMode.NESTED, nested_mode=NestedMode.WITHOUT_ROOT)
    parser.add_arguments(_run_args, dest="args")
    # Parse arguments and store in 'args'
    args = parser.parse_args().args
    # Check if 'camera_parameters' is not None and if so, generate the calibration file.
    if getattr(args, 'camera_parameters', None):  # using getattr to ensure backwards compatibility in case is_manual is not set
      temp_csv_path = _generate_temp_csv(args.camera_parameters)
      args.test.trajectory_via_calibration_file = temp_csv_path
    if getattr(args, 'object_center_addition', None): 
      additional = _generate_additional_center(args.object_center_addition)
      args.object_center_addition = additional
    return args
