from datasets import load_dataset, Dataset
import pathlib

DATA_DIR_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / "data"

train_dataset: Dataset = load_dataset("llm-book/wrime-sentiment", name="default", split="train")
valid_dataset: Dataset = load_dataset("llm-book/wrime-sentiment", name="default", split="validation")

train_dataset.to_csv(DATA_DIR_PATH / "train.csv")
valid_dataset.to_csv(DATA_DIR_PATH / "valid.csv")
