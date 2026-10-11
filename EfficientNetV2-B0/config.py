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

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.image_input_size <= 0:
            raise ValueError("image_input_size must be greater than 0")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.output_nodes <= 0:
            raise ValueError("output_nodes must be greater than 0")
        if self.transfer_learning_epochs <= 0:
            raise ValueError("transfer_learning_epochs must be greater than 0")
        if not 0 < self.unfreeze_fraction <= 1:
            raise ValueError("unfreeze_fraction must be greater than 0 and less than or equal to 1")
        if self.fine_tune_epochs <= 0:
            raise ValueError("fine_tune_epochs must be greater than 0")
        if self.fine_tune_lr <= 0:
            raise ValueError("fine_tune_lr must be greater than 0")

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.image_input_size, self.image_input_size, 3)


CONFIG = EfficientNetV2B0Config()
