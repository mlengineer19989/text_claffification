import pathlib

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DATA_DIR_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / "data"


my_features = Features(
    {"sentence": Value("string"), "label": ClassLabel(names=["positive", "negative"])}
)

# データを読み込む
df_train = pd.read_csv(DATA_DIR_PATH / "train.csv")
df_valid = pd.read_csv(DATA_DIR_PATH / "valid.csv")
train_dataset = Dataset.from_pandas(
    df_train[["sentence", "label"]], features=my_features
)
valid_dataset = Dataset.from_pandas(
    df_valid[["sentence", "label"]], features=my_features
)


##### トークナイズ #####
model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_text_classification(example: dict) -> BatchEncoding:
    """文書分類の事例のテキストをトークナイズし、IDに変換"""
    encoded_example = tokenizer(example["sentence"], max_length=512)
    # モデルの入力引数である"labels"をキーとして格納する
    encoded_example["labels"] = example["label"]
    return encoded_example


encoded_train_dataset = train_dataset.map(
    preprocess_text_classification,
    remove_columns=train_dataset.column_names,
)
encoded_valid_dataset = valid_dataset.map(
    preprocess_text_classification,
    remove_columns=valid_dataset.column_names,
)

##### ミニバッチ構築 #####
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch_inputs = data_collator(encoded_train_dataset[0:4])

##### モデル準備 #####
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
)

##### 訓練実行 #####
training_args = TrainingArguments(
    output_dir=str(pathlib.Path(__file__).parent / "outputs"),  # 結果の保存フォルダ
    per_device_train_batch_size=32,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=32,  # 評価時のバッチサイズ
    learning_rate=2e-5,  # 学習率
    lr_scheduler_type="linear",  # 学習率スケジューラの種類
    warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
    num_train_epochs=3,  # エポック数
    save_strategy="epoch",  # チェックポイントの保存タイミング
    logging_strategy="epoch",  # ロギングのタイミング
    evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
    load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
    metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
    # fp16=True,  # 自動混合精度演算の有効化
)


def compute_accuracy(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """予測ラベルと正解ラベルから正解率を計算"""
    predictions, labels = eval_pred
    # predictionsは各ラベルについてのスコア
    # 最もスコアの高いインデックスを予測ラベルとする
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


trainer = Trainer(
    model=model,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_accuracy,
)
trainer.train()
