__all__ = ["VoxCPM"]


def __getattr__(name):
    if name == "VoxCPM":
        from .core import VoxCPM

        return VoxCPM
    raise AttributeError(name)
