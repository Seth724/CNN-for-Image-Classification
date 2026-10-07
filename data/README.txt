-----------Genration of processed data------------

Put raw csv files here:
data/raw/fashion-mnist_train.csv
data/raw/fashion-mnist_test.csv

Run:
python scripts/prepare_data.py
python scripts/verify_data.py

Then team members can use this shared split data:
data/processed/fashion_mnist_70_15_15.npz
data/processed/split_manifest.csv
data/processed/metadata.json
