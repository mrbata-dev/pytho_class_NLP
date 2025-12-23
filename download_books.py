import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Book"],
    max_samples=100
)

dataset.export(
    export_dir="book_dataset",
    dataset_type=foz.types.YOLOv5Dataset
)

print("✅ 100 book images downloaded with labels")
