# Viettel AI Race - Medical NER & Assertion Pipeline

Đây là mã nguồn hệ thống trích xuất thực thể y khoa (Medical Named Entity Recognition) và phân loại ngữ cảnh (Assertion) dành cho văn bản y khoa tiếng Việt. 

Hệ thống cung cấp một End-to-End Pipeline có khả năng đọc văn bản y khoa thô, phát hiện các thực thể, xác định các ngữ cảnh (phủ định, tiền sử gia đình, tiền sử bệnh án) và ánh xạ chúng sang các mã chuẩn (ICD-10, RxNorm).

## Tính năng chính

- **Trích xuất thực thể (NER):** Nhận diện các thực thể như Tên bệnh (`CHẨN_ĐOÁN`), Triệu chứng (`TRIỆU_CHỨNG`), Thuốc (`THUỐC`), Tên xét nghiệm (`TÊN_XÉT_NGHIỆM`), Kết quả xét nghiệm (`KẾT_QUẢ_XÉT_NGHIỆM`) bằng mô hình `BamiBERT-ViMedNER`.
- **Phân loại ngữ cảnh (Assertion Classifier):** Xác định trạng thái của thực thể (`isNegated`, `isFamily`, `isHistorical`) thông qua 3 chế độ:
  - `rule_based`: (Mặc định) Dựa trên tập luật từ khóa.
  - `encoder`: Sử dụng mô hình phân loại HuggingFace đã được tinh chỉnh (Fine-tuned).
  - `llm`: Sử dụng Large Language Model (như `II-Medical-8B`) qua API.
- **Ánh xạ mã chuẩn (Semantic Linking):** So khớp thực thể bệnh với mã `ICD-10` và thuốc với mã `RxNorm` bằng công nghệ Semantic Search với `vihealthbert-base-word`.

## Yêu cầu môi trường

Dự án yêu cầu Python 3.8+ và các thư viện:
- `torch`
- `transformers`
- `sentence-transformers`
- `datasets`
- `evaluate`
- `underthesea`
- `pandas`, `numpy`, `scikit-learn`
- `accelerate` (dành cho huấn luyện)

Cài đặt các gói phụ thuộc (chạy từ thư mục gốc `code/`):
```bash
pip install -r requirements.txt
```

## Cấu trúc dự án mới

```
viettel_ai_race/code/
│
├── archives/               # Lưu trữ các file nén (ví dụ zip datasets, outputs)
├── data/                   # Quản lý tất cả dữ liệu
│   ├── input/              # Chứa các file văn bản y khoa gốc (.txt) làm đầu vào
│   ├── output/             # Chứa các file kết quả JSON, mô hình sau train
│   ├── raw/                # Dữ liệu thô (raw datasets)
│   └── clean/              # Các từ điển đã qua tiền xử lý (ICD-10, RxNorm)
├── logs/                   # Chứa các file nhật ký, ghi chú (diary, nb_dump)
├── notebooks/              # Chứa các file Jupyter Notebook dùng để finetune và EDA
├── src/                    # Mã nguồn chính của ứng dụng
│   ├── main.py             # File chạy hệ thống chính
│   ├── pipeline.py         # Định nghĩa các block xử lý (NER, Assertion, Linker)
│   ├── train_assertion.py  # Script huấn luyện mô hình Assertion
│   ├── preprocess_db.py    # Script tiền xử lý từ điển
│   └── ...                 # Các file xử lý và tiện ích khác
└── tools/                  # Công cụ hỗ trợ, ví dụ vncorenlp
```

## Hướng dẫn sử dụng

**LƯU Ý:** Do cấu trúc dự án đã được sắp xếp lại, tất cả các lệnh chạy code Python cần được thực thi từ bên trong thư mục `src/`.

### 1. Chuẩn bị dữ liệu đầu vào
Bạn hãy đặt các file văn bản y khoa (đuôi `.txt`) cần phân tích vào thư mục `data/input/input`.

### 2. Tiền xử lý từ điển (Tùy chọn)
Chạy script để chuẩn bị dữ liệu mã hóa ICD-10 và RxNorm:
```bash
cd src
python preprocess_db.py
```
Dữ liệu chuẩn sẽ được tạo tại `data/clean`.

### 3. Huấn luyện mô hình Assertion (Tùy chọn)
Nếu bạn muốn sử dụng mô hình Encoder để phân loại Assertion thay vì luật, bạn cần huấn luyện nó:
```bash
cd src
python train_assertion.py
```
Mô hình sẽ được lưu tại `data/output/model_assertion_output`.

### 4. Chạy Pipeline trích xuất
Để chạy toàn bộ hệ thống xử lý các file văn bản trong `data/input/input` và xuất ra `data/output/output`:
```bash
cd src
python main.py --assertion_mode <chế độ>
```
Các chế độ hỗ trợ: `rule_based` (mặc định), `encoder`, hoặc `llm`.

**Ví dụ chạy với mô hình Encoder:**
```bash
cd src
python main.py --assertion_mode encoder
```

Kết quả (định dạng JSON) sẽ xuất hiện trong thư mục `data/output/output/`.
