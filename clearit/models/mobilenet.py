from .encoder import BaseEncoder


class MobileNetEncoder(BaseEncoder):
    """MobileNet-backed encoder interface."""

    def __init__(self, output_size):
        super().__init__()

    def forward(self, x):
        pass
