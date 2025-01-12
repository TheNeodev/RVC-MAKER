import os
import shutil
import torch
import ctypes
import platform
import numpy as np
import tempfile as tf



model = torch.load(os.path.join("assets", "models", "predictors", "world.pth"), map_location="cpu")

if platform.system() == "Windows": model_type, suffix = ("world_64" if platform.architecture()[0] == "64bit" else "world_86"), ".dll"
else: model_type, suffix = "world_linux", ".so"

temp_folder = os.path.join("assets", "models", "predictors", "temp")
os.makedirs(temp_folder, exist_ok=True)

with tf.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_folder) as temp_file:
    temp_file.write(model[model_type])
    temp_path = temp_file.name

world_dll = ctypes.CDLL(temp_path)

class DioOption(ctypes.Structure):
    _fields_ = [("F0Floor", ctypes.c_double), ("F0Ceil", ctypes.c_double), ("ChannelsInOctave", ctypes.c_double), ("FramePeriod", ctypes.c_double), ("Speed", ctypes.c_int), ("AllowedRange", ctypes.c_double)]

class HarvestOption(ctypes.Structure):
    _fields_ = [("F0Floor", ctypes.c_double), ("F0Ceil", ctypes.c_double), ("FramePeriod", ctypes.c_double)]

def harvest(x, fs, f0_floor=50, f0_ceil=1100, frame_period=10):
    world_dll.Harvest.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(HarvestOption), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    world_dll.Harvest.restype = None 

    world_dll.InitializeHarvestOption.argtypes = [ctypes.POINTER(HarvestOption)]
    world_dll.InitializeHarvestOption.restype = None

    world_dll.GetSamplesForHarvest.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
    world_dll.GetSamplesForHarvest.restype = ctypes.c_int

    option = HarvestOption()
    world_dll.InitializeHarvestOption(ctypes.byref(option))

    option.F0Floor = f0_floor
    option.F0Ceil = f0_ceil
    option.FramePeriod = frame_period

    f0_length = world_dll.GetSamplesForHarvest(fs, len(x), option.FramePeriod)
    f0 = (ctypes.c_double * f0_length)()
    tpos = (ctypes.c_double * f0_length)()

    world_dll.Harvest((ctypes.c_double * len(x))(*x), len(x), fs, ctypes.byref(option), tpos, f0)
    return np.array(f0, dtype=np.float64), np.array(tpos, dtype=np.float64)

def dio(x, fs, f0_floor=50, f0_ceil=1100, channels_in_octave=2, frame_period=10, speed=1, allowed_range=0.1):
    world_dll.Dio.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(DioOption), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    world_dll.Dio.restype = None  

    world_dll.InitializeDioOption.argtypes = [ctypes.POINTER(DioOption)]
    world_dll.InitializeDioOption.restype = None

    world_dll.GetSamplesForDIO.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
    world_dll.GetSamplesForDIO.restype = ctypes.c_int

    option = DioOption()
    world_dll.InitializeDioOption(ctypes.byref(option))

    option.F0Floor = f0_floor
    option.F0Ceil = f0_ceil
    option.ChannelsInOctave = channels_in_octave
    option.FramePeriod = frame_period
    option.Speed = speed
    option.AllowedRange = allowed_range

    f0_length = world_dll.GetSamplesForDIO(fs, len(x), option.FramePeriod)
    f0 = (ctypes.c_double * f0_length)()
    tpos = (ctypes.c_double * f0_length)()

    world_dll.Dio((ctypes.c_double * len(x))(*x), len(x), fs, ctypes.byref(option), tpos, f0)
    return np.array(f0, dtype=np.float64), np.array(tpos, dtype=np.float64)

def stonemask(x, fs, tpos, f0):
    world_dll.StoneMask.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    world_dll.StoneMask.restype = None 

    out_f0 = (ctypes.c_double * len(f0))()
    world_dll.StoneMask((ctypes.c_double * len(x))(*x), len(x), fs, (ctypes.c_double * len(tpos))(*tpos), (ctypes.c_double * len(f0))(*f0), len(f0), out_f0)

    if os.path.exists(temp_folder): shutil.rmtree(temp_folder, ignore_errors=True)
    return np.array(out_f0, dtype=np.float64)