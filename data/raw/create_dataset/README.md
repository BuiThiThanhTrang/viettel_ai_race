# Medical NER Labeler – Hướng dẫn sử dụng

## Chạy ngay (khuyến nghị)

**Không cần cài đặt gì.** Chỉ cần:

1. Copy file **`Medical_NER_Labeler.html`** (trong thư mục `dist/`) sang máy cần dùng
2. Double-click để mở bằng trình duyệt (Chrome, Edge, Firefox đều được)
3. Dùng ngay, hoàn toàn offline

---

## Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 📂 Upload file | Kéo thả hoặc click nút Upload để tải file `.txt` |
| ✂️ Highlight | Bôi đen đoạn văn → tự động hiển thị text + vị trí |
| 🏷️ Gán nhãn | Chọn loại thực thể từ dropdown |
| 🔖 Assertions | Chọn isNegated / isFamily / isHistorical (chỉ cho TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC) |
| ➕ Thêm nhãn | Click "Thêm nhãn" hoặc nhấn `Ctrl+Enter` |
| 💾 Xuất JSON | Click "Xuất JSON" hoặc nhấn `Ctrl+S` |

## Các loại thực thể

- `TRIỆU_CHỨNG` — có assertions
- `TÊN_XÉT_NGHIỆM`
- `KẾT_QUẢ_XÉT_NGHIỆM`
- `CHẨN_ĐOÁN` — có assertions
- `THUỐC` — có assertions

## Định dạng JSON xuất ra

```json
[
  {
    "text": "ho đờm xanh",
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "position": [42, 53]
  },
  {
    "text": "viêm phổi",
    "type": "CHẨN_ĐOÁN",
    "assertions": ["isHistorical"],
    "position": [120, 129]
  }
]
```

## Build lại (nếu cần chỉnh sửa source)

Yêu cầu: Python 3.7+ (không cần cài thêm package nào)

```bash
cd code/create_dataset
python build.py
# Output: dist/Medical_NER_Labeler.html
```

## Cấu trúc thư mục

```
create_dataset/
├── index.html          # Source HTML
├── style.css           # Source CSS  
├── app.js              # Source JavaScript
├── build.py            # Script đóng gói thành 1 file
├── dist/
│   └── Medical_NER_Labeler.html   ← FILE NÀY để dùng
└── README.md
```
