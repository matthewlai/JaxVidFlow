import ctypes
import functools
import logging
import math
import os
import shutil
import subprocess
from typing import Optional, Sequence

import jax
from jax import numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# pub struct f0r_plugin_info {
#     /// < The (short) name of the plugin
#     pub name: *const ::std::os::raw::c_char,
#     /// < The plugin author
#     pub author: *const ::std::os::raw::c_char,
#     /// The plugin type, see PLUGIN_TYPE
#     pub plugin_type: ::std::os::raw::c_int,
#     /// < The color model used
#     pub color_model: ::std::os::raw::c_int,
#     /// < The frei0r major version this plugin is built for
#     pub frei0r_version: ::std::os::raw::c_int,
#     /// < The major version of the plugin
#     pub major_version: ::std::os::raw::c_int,
#     /// < The minor version of the plugin
#     pub minor_version: ::std::os::raw::c_int,
#     /// < The number of parameters of the plugin
#     pub num_params: ::std::os::raw::c_int,
#     /// < An optional explanation string
#     pub explanation: *const ::std::os::raw::c_char,
# }

class f0r_plugin_info(ctypes.Structure):
  _fields_ = [
    ('name', ctypes.c_char_p),
    ('author', ctypes.c_char_p),
    ('plugin_type', ctypes.c_int),
    ('color_model', ctypes.c_int),
    ('frei0r_version', ctypes.c_int),
    ('major_version', ctypes.c_int),
    ('minor_version', ctypes.c_int),
    ('num_params', ctypes.c_int),
    ('explanation', ctypes.c_char_p),
  ]

# pub struct f0r_param_info {
#     /// <The (short) name of the param
#     pub name: *const ::std::os::raw::c_char,
#     /// <The type (see the F0R_PARAM_* defines)
#     pub type_: ::std::os::raw::c_int,
#     /// <Optional explanation (can be 0)
#     pub explanation: *const ::std::os::raw::c_char,
# }

class f0r_param_info(ctypes.Structure):
  _fields_ = [
    ('name', ctypes.c_char_p),
    ('type_', ctypes.c_int),
    ('explanation', ctypes.c_char_p),
  ]

def _debug_print_structure(s):
  for field in s._fields_:
    print(f'{field[0]}: {getattr(s, field[0])}')

@functools.cache
def _find_gyroflow(suggestion: str | None = None) -> Optional[str]:
  candidates = [
    # MacOS.
    '/Applications/Gyroflow.app/Contents/MacOS/gyroflow',

    # In PATH.
    'gyroflow'

    # TODO: What do we do for Windows?
  ]

  if suggestion is not None:
    candidates = [suggestion] + candidates

  for candidate in candidates:
    candidate = shutil.which(candidate)
    if os.path.exists(candidate):
      logger.info(f'Gyroflow binary found at: {candidate}')
      return candidate
  return None

@functools.cache
def _load_lib(path: str):
  lib = ctypes.CDLL(path)
  lib.f0r_get_plugin_info.argtypes = [ctypes.POINTER(f0r_plugin_info)]
  lib.f0r_get_plugin_info.restype = None
  lib.f0r_get_param_info.argtypes = [ctypes.POINTER(f0r_param_info), ctypes.c_int]
  lib.f0r_get_param_info.restype = None
  lib.f0r_construct.argtypes = [ctypes.c_uint, ctypes.c_uint]
  lib.f0r_construct.restype = ctypes.c_void_p
  lib.f0r_destruct.argtypes = [ctypes.c_void_p]
  lib.f0r_destruct.restype = None
  lib.f0r_set_param_value.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
  lib.f0r_set_param_value.restype = None
  lib.f0r_update.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.POINTER(ctypes.c_uint32),
                             ctypes.POINTER(ctypes.c_uint32)]
  lib.f0r_update.restype = None

  assert lib.f0r_init() == 1, "Gyroflow plugin doesn't seem to have been loaded correctly"

  return lib

def gyroflow_create_project_file(video_path: str, preset: str = '',
                                 gyroflow_path: str | None = None) -> str:
  gf_path = _find_gyroflow(suggestion=gyroflow_path)
  # Gyroflow seems to only work with absolute paths.
  logger.info('Gyroflow analysing motion')
  video_path = os.path.abspath(video_path)
  args = [gf_path, video_path, '--export-project', '3']
  if preset:
    args.extend(['--preset', preset])
  try:
    res = subprocess.run(args, capture_output=True, encoding='utf-8', timeout=10.0)
  except subprocess.TimeoutExpired as e:
    logger.error(f'Gyroflow timeout, output:')
    logger.error(e.output.decode('utf-8'))
  assert res.returncode == 0, f'Gyroflow returned {res.returncode}'

  root, ext = os.path.splitext(video_path)
  gyroflow_project_path = root + '.gyroflow'
  assert os.path.exists(gyroflow_project_path)
  logger.info('Gyroflow analysis done')
  return gyroflow_project_path

@jax.jit
def to_gyroflow(frame: jnp.ndarray) -> jnp.ndarray:
  # Gyroflow requires uint8 RGBA packed.
  frame = (frame * 255).astype(jnp.uint8)
  return jnp.pad(frame, pad_width=((0, 0), (0, 0), (0, 1)), constant_values=255)

@jax.jit
def from_gyroflow(frame: jnp.ndarray) -> jnp.ndarray:
  return frame[:, :, :3].astype(jnp.float32) / 255.0

class Gyroflow:
  def __init__(self, gyroflow_project_path: str, gyroflow_lib_path: str | None = None):
    self._lib = _load_lib(gyroflow_lib_path)

    plugin_info = f0r_plugin_info()
    self._lib.f0r_get_plugin_info(plugin_info)
    assert plugin_info.plugin_type == 0  # Filter
    assert plugin_info.color_model == 2 # Packed 32
    assert plugin_info.frei0r_version == 1

    self._param_infos = []

    for param_id in range(plugin_info.num_params):
      param_info = f0r_param_info()
      self._lib.f0r_get_param_info(param_info, param_id)
      name = param_info.name.decode('utf-8')
      self._param_infos.append((name, param_info))
    self._instance = None
    self._project_path = gyroflow_project_path

    # We delay by one frame to avoid unnecessary GPU sync.
    self._last_frame = None
    self._last_frame_time = None

  def _find_param_index(self, name) -> int:
    for i, (param_name, _) in enumerate(self._param_infos):
      if param_name == name:
        return i
    raise ValueError(f'Unknown param: {name}')

  def process_frame(self, frame: jnp.ndarray | None, frame_time: float | None, rotation: int) -> jnp.ndarray | None:
    if frame is not None:
      assert frame.shape[2] == 4 and frame.dtype == jnp.uint8, 'Gyroflow expects RGBA packed. Use gyroflow.to_gyroflow()/from_gyroflow() to convert.'
      assert frame_time is not None

    if self._instance is None:
      assert frame is not None
      height, width = frame.shape[:2]
      self._instance = self._lib.f0r_construct(width, height)
      path_bytes = self._project_path.encode('utf-8')
      ArgType = ctypes.c_char_p * 1
      arg = ArgType()
      arg[0] = path_bytes
      self._lib.f0r_set_param_value.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
      self._lib.f0r_set_param_value(self._instance, arg, self._find_param_index('Project'))

    ret = None

    if self._last_frame is not None:
      height, width = self._last_frame.shape[:2]
      OutFrameType = ctypes.c_uint32 * (width * height)
      out_frame = OutFrameType()
      in_frame = ctypes.cast(self._last_frame.tobytes(), ctypes.POINTER(ctypes.c_uint32))
      self._lib.f0r_update(self._instance, self._last_frame_time, in_frame, out_frame)

      np_out_frame = np.ctypeslib.as_array(out_frame).view(np.uint8).reshape((height, width, 4))

      ret = jnp.asarray(np_out_frame, device=frame.device)
    else:
      ret = None

    self._last_frame = frame
    self._last_frame_time = frame_time

    return ret

  def __del__(self):
    if self._instance is not None:
      self._lib.f0r_destruct(self._instance)
