from pathlib import Path
from typing import Any

import tensorflow as tf


def _shape_dimension(value: Any) -> int:
    if value is None:
        raise ValueError("MAC calculation requires concrete layer output shapes")
    return int(value)


def count_macs(model: tf.keras.Model) -> tuple[int, list[dict[str, Any]]]:
    """Count convolution and dense MACs and return layer-level details.

    Bias additions, pooling, activations, and global-average-pooling reductions
    are excluded so the result follows the assignment's formulas.
    """
    details = []
    total = 0
    for layer in model.layers:
        params = int(layer.count_params())
        macs = 0
        output_shape = layer.output.shape

        if isinstance(layer, tf.keras.layers.Conv2D):
            height = _shape_dimension(output_shape[1])
            width = _shape_dimension(output_shape[2])
            input_channels = _shape_dimension(layer.input.shape[-1])
            kernel_height, kernel_width, _, output_channels = layer.kernel.shape
            macs = height * width * int(kernel_height) * int(kernel_width) * input_channels * int(output_channels)
        elif isinstance(layer, tf.keras.layers.Dense):
            input_units = _shape_dimension(layer.input.shape[-1])
            output_units = _shape_dimension(output_shape[-1])
            macs = input_units * output_units

        total += macs
        details.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": [None if value is None else int(value) for value in output_shape],
                "parameters": params,
                "macs": macs,
            }
        )
    return total, details


def architecture_metrics(model: tf.keras.Model) -> dict[str, Any]:
    total_macs, layers = count_macs(model)
    return {
        "model": model.name,
        "total_parameters": int(model.count_params()),
        "total_macs": total_macs,
        "layers": layers,
    }


def save_json(data: dict[str, Any], path: str | Path) -> None:
    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")