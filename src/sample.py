from transformers import AutoTokenizer

model_name: str = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoded_input = tokenizer("これはテストです。")
print(encoded_input)

