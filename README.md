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

Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 1. Huấn luyện mô hình Assertion (Tùy chọn)
Nếu bạn muốn sử dụng mô hình Encoder để phân loại Assertion, bạn cần huấn luyện nó trước tiên:
```bash
python train_assertion.py
```
Quá trình này sẽ tạo ra thư mục `model_assertion_output` chứa mô hình và tokenizer.

### 2. Chạy Pipeline trích xuất
Để chạy toàn bộ hệ thống xử lý các file văn bản (đuôi `.txt`) trong thư mục `input`:
```bash
python main.py --assertion_mode <chế độ>
```
Các chế độ hỗ trợ: `rule_based` (mặc định), `encoder`, hoặc `llm`.

**Ví dụ chạy với mô hình Encoder:**
```bash
python main.py --assertion_mode encoder
```

Kết quả (định dạng JSON) sẽ được lưu trong thư mục `output`.

## Cấu trúc thư mục

```
viettel_ai_race/
│
├── data/                   # Chứa dữ liệu từ điển mã hóa (ICD-10, RxNorm)
├── input/                  # Chứa các file văn bản y khoa gốc (.txt)
├── output/                 # Chứa các kết quả JSON sau khi trích xuất
│
├── pipeline.py             # Mã nguồn chính định nghĩa các Pipeline, NER, Assertion và Linker
├── main.py                 # File chạy chính để xử lý hàng loạt văn bản
├── train_assertion.py      # Script huấn luyện mô hình Assertion (Encoder)
├── utils.py                # Các hàm tiện ích hỗ trợ
└── requirements.txt        # Danh sách các thư viện yêu cầu
```
