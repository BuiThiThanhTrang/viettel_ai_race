import json
import os

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in text.split('\n')]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in text.split('\n')]
    })

add_markdown("# BamiBERT ViMedNER - 2 Stage Finetuning (V3)\nBản cập nhật v3: Giải quyết vấn đề Catastrophic Forgetting bằng cách bảo toàn nguyên vẹn Classification Head của BamiBERT gốc sau khi trải qua Stage 1 (MLM).")

add_code("""# ============================================================
# CELL 1 - Cài đặt thư viện
# ============================================================
%uv pip install --upgrade torch torchvision transformers==5.5.0 datasets==4.3.0 evaluate accelerate huggingface_hub seqeval
%uv pip install underthesea

print("✅ Cài đặt hoàn tất!")""")

add_code("""!mkdir -p /root/data/train
!mkdir -p /root/data/finetune/text
!mkdir -p /root/data/finetune/gt""")

add_code("""# ============================================================
# CELL 2 - Cấu hình: HF Token, Repo, Đường dẫn dữ liệu
# ============================================================
import os

# ⚠️ ĐIỀN TOKEN CỦA BẠN VÀO ĐÂY
HF_TOKEN = "YOUR_HF_TOKEN_HERE"

HF_REPO_NAME = "BamiBERT-ViMedNER-Finetuned-v3"
BASE_MODEL_NAME = "cbc-528a/BamiBERT-ViMedNER"

DATA_BASE         = "/root/data"
TRAIN_TEXT_DIR    = os.path.join(DATA_BASE, "train")              # 400 files txt không nhãn
FINETUNE_TEXT_DIR = os.path.join(DATA_BASE, "finetune", "text")   # 48 files txt có nhãn
FINETUNE_GT_DIR   = os.path.join(DATA_BASE, "finetune", "gt")     # 48 files json GT

STAGE1_OUTPUT = "/root/bami_adapted_mlm"
STAGE2_OUTPUT = "/root/bami_ner_finetuned"

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 20
BATCH_SIZE    = 8
MAX_SEQ_LEN   = 1024

print("✅ Cấu hình hoàn tất!")""")

add_code("""# ============================================================
# CELL 3 (MỚI) - CẤT GIỮ CLASSIFICATION HEAD GỐC
# Bước này cực kỳ quan trọng để không bị mất khả năng NER
# ============================================================
import torch
from transformers import AutoModelForTokenClassification

print(f"🧰 Đang tải mô hình {BASE_MODEL_NAME} để tháo Classification Head...")
orig_ner_model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL_NAME)

saved_head = {
    "weight": orig_ner_model.classifier.weight.clone().detach(),
    "bias": orig_ner_model.classifier.bias.clone().detach()
}

torch.save(saved_head, "/root/saved_ner_head.pt")
print("✅ Đã cất giữ thành công vào /root/saved_ner_head.pt!")

# Xóa để giải phóng RAM
del orig_ner_model""")

add_code("""# ============================================================
# CELL 4 - Stage 1: Continued MLM Pretraining
# ============================================================
import glob
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# 2. Load unlabeled docs
train_txt_files = sorted(glob.glob(os.path.join(TRAIN_TEXT_DIR, "*.txt")))
raw_texts = []
for fpath in train_txt_files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content:
            raw_texts.append(content)

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LEN)

raw_dataset = Dataset.from_dict({"text": raw_texts})
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator_mlm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 4. Load model MLM
model_mlm = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_NAME)

stage1_args = TrainingArguments(
    output_dir=STAGE1_OUTPUT,
    num_train_epochs=STAGE1_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    save_strategy="epoch",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    report_to="none",
    push_to_hub=False,
)

trainer_mlm = Trainer(
    model=model_mlm,
    args=stage1_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator_mlm,
    processing_class=tokenizer,
)

print(f"\\n🚀 [Stage 1] Bắt đầu Continued MLM Pretraining...")
trainer_mlm.train()
trainer_mlm.save_model(STAGE1_OUTPUT)
tokenizer.save_pretrained(STAGE1_OUTPUT)
print("✅ [Stage 1] Hoàn tất!")""")


add_code("""# ============================================================
# CELL 5 - Chuẩn bị BIO Dataset (Gom từ giữ độ dài)
# ============================================================
import json
import random
import glob
import os
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast
from underthesea import word_tokenize

tokenizer = PreTrainedTokenizerFast.from_pretrained(STAGE1_OUTPUT)

def segment_text_preserve_length(text):
    tokens = word_tokenize(text)
    result = list(text)
    search_idx = 0
    for token in tokens:
        if ' ' in token:
            pos = text.find(token, search_idx)
            if pos != -1:
                for i in range(pos, pos + len(token)):
                    if result[i] == ' ':
                        result[i] = '_'
                search_idx = pos + len(token)
    return ''.join(result)

LABEL_MAPPING = {
    "CHẨN_ĐOÁN": "ten_benh",
    "THUỐC": "bien_phap_dieu_tri",
    "TRIỆU_CHỨNG": "trieu_chung_benh",
    "TÊN_XÉT_NGHIỆM": "bien_phap_chan_doan",
    "KẾT_QUẢ_XÉT_NGHIỆM": "nguyen_nhan_benh"
}
ENTITY_TYPES = list(LABEL_MAPPING.keys())

label2id = {
    'B-bien_phap_chan_doan': 0, 'B-bien_phap_dieu_tri': 1, 'B-nguyen_nhan_benh': 2, 'B-ten_benh': 3, 'B-trieu_chung_benh': 4,
    'I-bien_phap_chan_doan': 5, 'I-bien_phap_dieu_tri': 6, 'I-nguyen_nhan_benh': 7, 'I-ten_benh': 8, 'I-trieu_chung_benh': 9,
    'O': 10
}
id2label = {v: k for k, v in label2id.items()}
LABELS = list(label2id.keys())

def convert_gt_to_bio_tokens(text, entities, tokenizer, max_length=512):
    char_labels = ["O"] * len(text)
    for ent in entities:
        pos, etype = ent.get("position"), ent.get("type", "")
        if not pos or len(pos) < 2 or etype not in ENTITY_TYPES: continue
        start, end = pos[0], pos[1]
        if start >= len(text) or end > len(text) or start >= end: continue
        
        native_etype = LABEL_MAPPING[etype]
        char_labels[start] = f"B-{native_etype}"
        for ci in range(start + 1, end):
            char_labels[ci] = f"I-{native_etype}"

    encoding = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True, padding="max_length")

    token_labels = []
    for tok_start, tok_end in encoding["offset_mapping"]:
        if tok_start == tok_end: token_labels.append(-100)
        else: token_labels.append(label2id[char_labels[tok_start]])

    return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"], "labels": token_labels}

gt_files = sorted(glob.glob(os.path.join(FINETUNE_GT_DIR, "*.json")))
all_samples = []

for gt_path in gt_files:
    stem = os.path.basename(gt_path).replace("_labels", "").replace(".json", "")
    txt_path = os.path.join(FINETUNE_TEXT_DIR, f"{stem}.txt")
    if not os.path.exists(txt_path): continue

    with open(txt_path, "r", encoding="utf-8") as f: text = f.read().strip()
    with open(gt_path, "r", encoding="utf-8") as f: entities = json.load(f)
    if not text or not entities: continue

    text = segment_text_preserve_length(text)
    all_samples.append(convert_gt_to_bio_tokens(text, entities, tokenizer, MAX_SEQ_LEN))

random.seed(42)
random.shuffle(all_samples)
val_size = max(1, int(len(all_samples) * 0.1))

val_samples = all_samples[:val_size]
train_samples = all_samples[val_size:]

def to_hf_dataset(samples):
    return Dataset.from_dict({k: [s[k] for s in samples] for k in ["input_ids", "attention_mask", "labels"]})

bio_dataset = DatasetDict({"train": to_hf_dataset(train_samples), "validation": to_hf_dataset(val_samples)})
print(f"✅ Dataset sẵn sàng: {len(train_samples)} Train, {len(val_samples)} Val")""")


add_code("""# ============================================================
# CELL 6 - Stage 2: NER Fine-tuning với Head Restoration
# ============================================================
import evaluate
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

model_ner = AutoModelForTokenClassification.from_pretrained(
    STAGE1_OUTPUT,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# KHÔI PHỤC LẠI TRÍ NHỚ (CLASSIFICATION HEAD CŨ)
print("🔧 Đang lắp lại Classification Head cũ để khôi phục trí nhớ...")
saved_head = torch.load("/root/saved_ner_head.pt", weights_only=True)
with torch.no_grad():
    model_ner.classifier.weight.copy_(saved_head["weight"])
    model_ner.classifier.bias.copy_(saved_head["bias"])
print("✅ Đã lắp ráp thành công!")

# ĐÓNG BĂNG 10 TẦNG DƯỚI
for param in model_ner.roberta.embeddings.parameters(): param.requires_grad = False
for i in range(10):
    for param in model_ner.roberta.encoder.layer[i].parameters(): param.requires_grad = False
print("✅ Đã đóng băng embeddings và 10 lớp encoder dưới cùng!")

data_collator_ner = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)
seqeval_metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_labels_batch, pred_labels_batch = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        t, p = [], []
        for pr, la in zip(pred_seq, label_seq):
            if la != -100:
                t.append(id2label[la])
                p.append(id2label[pr])
        true_labels_batch.append(t)
        pred_labels_batch.append(p)
    results = seqeval_metric.compute(predictions=pred_labels_batch, references=true_labels_batch)
    return {"precision": results.get("overall_precision", 0), "recall": results.get("overall_recall", 0), "f1": results.get("overall_f1", 0)}

stage2_args = TrainingArguments(
    output_dir=STAGE2_OUTPUT,
    num_train_epochs=STAGE2_EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    logging_steps=5,
    report_to="none",
)

trainer_ner = Trainer(
    model=model_ner,
    args=stage2_args,
    train_dataset=bio_dataset["train"],
    eval_dataset=bio_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator_ner,
    compute_metrics=compute_metrics,
)

print(f"\\n🚀 Bắt đầu NER Fine-tuning...")
trainer_ner.train()
trainer_ner.save_model(STAGE2_OUTPUT)
tokenizer.save_pretrained(STAGE2_OUTPUT)
print(f"\\n✅ Đã lưu model hoàn chỉnh tại: {STAGE2_OUTPUT}")""")

add_code("""# ============================================================
# CELL 7 - Đánh giá
# ============================================================
eval_results = trainer_ner.evaluate()
print("\\n📊 Kết quả Validation:")
print(f"  F1: {eval_results.get('eval_f1', 0):.4f}")
print(f"  P:  {eval_results.get('eval_precision', 0):.4f}")
print(f"  R:  {eval_results.get('eval_recall', 0):.4f}")""")


add_code("""# ============================================================
# CELL 8 - Upload Hub
# ============================================================
from huggingface_hub import HfApi, login
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast

login(token=HF_TOKEN)
api = HfApi()
user_info = api.whoami(token=HF_TOKEN)
FULL_REPO_ID = f"{user_info['name']}/{HF_REPO_NAME}"

print(f"📦 Tạo repo: {FULL_REPO_ID}")
api.create_repo(repo_id=FULL_REPO_ID, repo_type="model", exist_ok=True, private=False)

final_model = AutoModelForTokenClassification.from_pretrained(STAGE2_OUTPUT)
final_tokenizer = PreTrainedTokenizerFast.from_pretrained(STAGE2_OUTPUT)

final_model.push_to_hub(FULL_REPO_ID)
final_tokenizer.push_to_hub(FULL_REPO_ID)
print("🎉 Đã đẩy model lên HF thành công!")""")

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("finetune_ner_v3.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Notebook generated successfully!")
