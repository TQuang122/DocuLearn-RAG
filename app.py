import spaces
from src.ui import build_demo, launch_demo


@spaces.GPU
def _zero_gpu_provider():
    """Satisfy ZeroGPU hardware requirement.
    The actual Gradio UI runs in the main process (no lambdas to pickle)."""
    return None


demo = build_demo()

if __name__ == "__main__":
    launch_demo(demo)
