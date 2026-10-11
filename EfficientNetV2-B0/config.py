from dataclasses import dataclass


@dataclass(frozen=True)
class EfficientNetV2B0Config:
    batch_size: int = 32
    image_input_size: int = 32
    dropout: float = 0.2
    output_nodes: int = 10
    activation: str = "softmax"
    transfer_learning_epochs: int = 10
    unfreeze_fraction: float = 0.3
    fine_tune_epochs: int = 20
    fine_tune_lr: float = 1e-5

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.image_input_size, self.image_input_size, 3)


CONFIG = EfficientNetV2B0Config()
