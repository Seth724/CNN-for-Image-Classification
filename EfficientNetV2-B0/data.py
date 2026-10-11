import sys
import pathlib
from tensorflow import data as tf_data

# Make the repo root importable so `src` resolves when running this file directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data_loader import make_tf_datasets
import matplotlib.pyplot as plt

train_ds, val_ds, test_ds = make_tf_datasets(batch_size=1, sota=True)

print("Train dataset size", train_ds.cardinality().numpy())
print("Validation dataset size", val_ds.cardinality().numpy())
print("Test dataset size", test_ds.cardinality().numpy())

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[0].numpy().astype("uint8"))
    plt.title(int(label[0]))
    plt.axis("off")

plt.show()
