import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
import os

# 1. Định nghĩa tập nhãn (Labels)
# Để đơn giản hóa, ta coi bài toán là Multi-class classification (mỗi câu chỉ thuộc 1 nhãn assertion)
# Nếu thực tế 1 câu có thể mang nhiều nhãn, cần dùng cấu trúc Multi-label (BCEWithLogitsLoss).
# Ở đây ta demo Multi-class.
labels = ["None", "isNegated", "isFamily", "isHistorical"]
label2id = {lbl: i for i, lbl in enumerate(labels)}
id2label = {i: lbl for i, lbl in enumerate(labels)}

# 2. Tạo dữ liệu giả lập (Dummy Data)
# Trong thực tế, bạn cần chuẩn bị một file CSV chứa 2 cột: text và label
dummy_data = [
    {"text": "Bệnh nhân có tiền sử [E] tăng huyết áp [/E] từ 10 năm trước.", "label": "isHistorical"},
    {"text": "Người nhà chưa ghi nhận ai bị [E] đái tháo đường [/E].", "label": "isFamily"},
    {"text": "Bệnh nhân hoàn toàn tỉnh táo, không [E] đau ngực [/E].", "label": "isNegated"},
    {"text": "Khám lâm sàng phát hiện [E] gan to [/E] dưới bờ sườn 2cm.", "label": "None"},
    {"text": "Bố ruột mất vì [E] ung thư phổi [/E] năm 60 tuổi.", "label": "isFamily"},
    {"text": "Phủ nhận [E] dị ứng thuốc [/E].", "label": "isNegated"},
]

df = pd.DataFrame(dummy_data)
df['label'] = df['label'].map(label2id)

# Chuyển đổi DataFrame thành Dataset của HuggingFace
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 3. Khởi tạo Tokenizer
model_name = "demdecuong/vihealthbert-base-word"
print(f"Đang tải Tokenizer từ {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Thêm 2 thẻ tag đặc biệt [E] và [/E] vào từ vựng của tokenizer
special_tokens_dict = {'additional_special_tokens': ['[E]', '[/E]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Đã thêm {num_added_toks} token đặc biệt.")

# Hàm tiền xử lý (Tokenization)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. Khởi tạo Model
print(f"Đang tải Model từ {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(labels), 
    id2label=id2label, 
    label2id=label2id
)

# Rất quan trọng: Phải thay đổi kích thước Embedding layer vì ta vừa thêm token mới
model.resize_token_embeddings(len(tokenizer))

# 5. Cấu hình Training
training_args = TrainingArguments(
    output_dir="./..\data\output\model_assertion_output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# 6. Viết hàm tính độ chính xác (Metrics)
import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 7. Khởi tạo Trainer và Huấn luyện
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Bắt đầu huấn luyện...")
trainer.train()

# 8. Lưu model sau khi huấn luyện xong
final_model_path = "./..\data\output\model_assertion_output"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Huấn luyện thành công. Model đã được lưu tại: {final_model_path}")
