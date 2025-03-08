import os
import yaml
import logging
import traceback
from functools import lru_cache
from contextlib import contextmanager
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """
    Load and return the YAML config file as a dictionary
    
    Args:
        config_path (str): Path to the YAML config file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise

def ensure_dir(directory):
    """
    Ensure that a directory exists; if it doesn't, create it
    
    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def get_device():
    """
    Get the appropriate device for PyTorch, with preference for MPS on M3 Pro
    but falling back to CPU for memory-intensive operations
    
    Returns:
        torch.device: The device to use (mps, cuda, or cpu)
    """
    # Check environment variable to force CPU usage
    if os.environ.get("FORCE_CPU", "0") == "1":
        logger.info("Forcing CPU usage as specified by environment variable")
        return torch.device("cpu")
    
    # Check available memory on MPS
    if torch.backends.mps.is_available():
        try:
            # Try importing psutil to check memory
            import psutil
            
            # Get system memory info
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            
            # If less than 4GB available, use CPU to avoid OOM
            if available_gb < 4:
                logger.warning(f"Low memory available ({available_gb:.1f} GB). Using CPU to avoid OOM errors")
                return torch.device("cpu")
                
            logger.info(f"Using MPS (Metal Performance Shaders) device with {available_gb:.1f} GB available memory")
            return torch.device("mps")
        except ImportError:
            # If psutil not available, proceed with MPS but log a warning
            logger.warning("psutil not installed; cannot check available memory. Using MPS but may encounter OOM errors")
            return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")

@lru_cache(maxsize=1000)
def cache_function(func):
    """
    Decorator that caches the results of a function call.
    
    Args:
        func: The function to cache
        
    Returns:
        The decorated function with caching
    """
    return func

class MPSGradScaler:
    """
    Gradient scaler for MPS devices to enable mixed precision training.
    
    This is a simplified version of torch.cuda.amp.GradScaler adapted for MPS.
    It scales the loss to prevent underflow in float16 gradients during backpropagation.
    
    Usage:
    ```
    # In your training loop:
    scaler = MPSGradScaler(enabled=use_mps_mixed_precision)
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass in mixed precision
        with MPSAutocast(enabled=use_mps_mixed_precision):
            outputs = model(**inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        
        # Step with unscaled gradients
        scaler.step(optimizer)
        
        # Update the scale for next iteration
        scaler.update()
    ```
    
    Args:
        enabled (bool): Whether to use gradient scaling
        init_scale (float): Initial scale factor
        growth_factor (float): Factor by which to increase scale after successful steps
        backoff_factor (float): Factor by which to decrease scale after NaN/inf gradients
        growth_interval (int): Number of consecutive successful steps before increasing scale
    """
    def __init__(self, enabled=True, init_scale=128.0, growth_factor=2.0, 
                 backoff_factor=0.5, growth_interval=2000):
        self.enabled = enabled
        self.scale_factor = init_scale  # Renamed from 'scale' to avoid method name conflict
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.inf_or_nan = False
        self._growth_tracker = 0
        
    def scale(self, loss):
        """Scale the loss by the scale factor."""
        if self.enabled:
            return loss * self.scale_factor
        return loss
    
    def unscale_(self, optimizer):
        """Unscale gradients contained in optimizer."""
        if not self.enabled:
            return
        
        # Check for inf/NaN values
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if torch.isfinite(param.grad).all():
                        param.grad.div_(self.scale_factor)
                    else:
                        self.inf_or_nan = True
                        param.grad = None  # Set NaN gradients to None
    
    def step(self, optimizer):
        """Unscale and step optimizer."""
        if not self.enabled:
            return optimizer.step()
            
        self.unscale_(optimizer)
        
        if not self.inf_or_nan:
            return optimizer.step()
        else:
            logger.warning("Skipping optimizer step due to inf/NaN gradients")
            return None
    
    def update(self):
        """Update the scale factor."""
        if not self.enabled:
            return
            
        if not self.inf_or_nan:
            # If no NaN/inf detected, increase consecutive success counter
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale_factor *= self.growth_factor
                self._growth_tracker = 0
                logger.debug(f"Increasing gradient scale to {self.scale_factor}")
        else:
            # If NaN/inf detected, reduce scale and reset counter
            self.scale_factor *= self.backoff_factor
            self._growth_tracker = 0
            logger.warning(f"Gradient overflow detected, decreasing scale to {self.scale_factor}")
            self.inf_or_nan = False

@contextmanager
def MPSAutocast(enabled=True):
    """
    Context manager for MPS mixed precision training.
    
    This context manager temporarily sets the default dtype to float16 for mixed precision
    training on MPS devices, similar to torch.cuda.amp.autocast but adapted for MPS.
    
    Usage:
    ```
    # Configure from config.yaml
    use_mixed_precision = config['training']['mps_mixed_precision']
    
    # In your training loop
    with MPSAutocast(enabled=use_mixed_precision):
        outputs = model(**inputs)
        loss = criterion(outputs, targets)
    ```
    
    Args:
        enabled (bool): Whether to enable reduced precision
    """
    if enabled and torch.backends.mps.is_available():
        orig_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            yield
        finally:
            torch.set_default_dtype(orig_dtype)
    else:
        yield

def configure_mps_memory(mps_memory_efficient=True):
    """
    Configure MPS memory settings for optimal performance.
    
    This function sets environment variables that help manage memory usage
    on Apple Silicon (M1/M2/M3) devices when using PyTorch's MPS backend.
    
    Args:
        mps_memory_efficient (bool): Whether to enable memory-efficient MPS settings
    """
    if torch.backends.mps.is_available() and mps_memory_efficient:
        # These environment variables help with MPS memory management
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        logger.info("MPS memory efficiency settings enabled")
        
    return