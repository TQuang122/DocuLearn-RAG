import spaces
from src.ui import build_demo, launch_demo


@spaces.GPU
def _build_demo():
    return build_demo()


demo = _build_demo()

if __name__ == "__main__":
    launch_demo(demo)
