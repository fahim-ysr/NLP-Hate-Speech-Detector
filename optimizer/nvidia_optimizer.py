# Importing required modules

import os
import tensorflow as tf
from tensorflow.keras import mixed_precision


def optimize_gpu():
    """
    Configures GPU for optimal performance: Enables dynamic memory allocation (prevents OOM crashes) and mixed precision (2-3x faster on RTX cards)
    """

    # Force TensorFlow to only see NVIDIA GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only the first CUDA-capable GPU (NVIDIA)

    # Disable oneDNN optimizations that might cause issues
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Enable dynamic memory growth (prevents TensorFlow from hogging all GPU memory)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Only use the first GPU (NVIDIA RTX 3050 Ti)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Enable memory growth (prevents TensorFlow from allocating all VRAM at once)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Verify which GPU is being used
            print(f"Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            print(f"Using GPU: {gpus[0].name}")
            print(f"Enabled memory growth")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found! Training will be very slow on CPU.")
        return False
    
    # Enable mixed precision for faster training
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Enabled mixed precision (float16) - up to 3x faster on RTX GPUs")
    
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
    
    # Print memory info
    try:
        gpu_devices = tf.config.list_logical_devices('GPU')
        if gpu_devices:
            print(f"Logical GPU devices: {len(gpu_devices)}")
    
    except:
        pass
    
    return True


def get_gpu_memory_info():
    """
    Returns current GPU memory usage (for debugging)
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("\nGPU Memory Status:")
            print(result.stdout)
            return result.stdout
    except Exception as e:
        print(f"Could not get GPU info: {e}")
    return None