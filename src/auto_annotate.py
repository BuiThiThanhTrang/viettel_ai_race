import google.generativeai as genai
import os
import glob
import time
import re
import json

# =======================================================
# ⚠️ CẤU HÌNH API KEY CỦA GEMINI Ở ĐÂY
# Bạn có thể lấy API Key miễn phí tại: https://aistudio.google.com/app/apikey
# =======================================================
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)

# Sử dụng model Gemini 1.5 Pro (hoặc 1.5 Flash nếu muốn nhanh/rẻ hơn)
model = genai.GenerativeModel('gemini-3.5-flash-lite')

PROMPT_TEMPLATE = """
Bạn là một chuyên gia gán nhãn dữ liệu y khoa (Medical Annotator) bằng tiếng Việt. Nhiệm vụ của bạn là đọc các đoạn văn bản lâm sàng và bọc các thực thể y tế (Medical Entities) bằng các thẻ XML tương ứng.

[CÁC LOẠI THẺ XML VÀ ĐỊNH NGHĨA]
1. <trieu_chung> ... </trieu_chung>: Biểu hiện, cảm giác hoặc dấu hiệu lâm sàng của bệnh nhân (VD: sốt, đau bụng, buồn nôn, nhịp tim nhanh).
2. <ten_xet_nghiem> ... </ten_xet_nghiem>: Tên các xét nghiệm, thăm dò, chẩn đoán hình ảnh (VD: công thức máu, AST, siêu âm bụng, nội soi).
3. <ket_qua_xet_nghiem> ... </ket_qua_xet_nghiem>: Giá trị định lượng (kèm đơn vị), định tính hoặc mô tả hình ảnh (VD: 421, dương tính, túi mật to nhiều).
4. <chan_doan> ... </chan_doan>: Tên bệnh, hội chứng, tổn thương hoặc tình trạng bệnh lý (VD: viêm ruột do virus, loét thực quản độ C, suy thận cấp).
5. <thuoc> ... </thuoc>: Tên thuốc, bao gồm cả hàm lượng, đường dùng, tần suất (VD: amlodipine 10 mg po daily, omeprazole).

[QUY TẮC GÁN NHÃN NGHIÊM NGẶT - BẮT BUỘC TUÂN THỦ]
- Quy tắc Span: Phải bọc CHÍNH XÁC cụm từ. GIỮ LẠI các từ bổ nghĩa trực tiếp (mức độ, vị trí, tính chất).
- Lọc bỏ từ dẫn triệu chứng: KHÔNG bọc các từ dẫn như "cảm giác", "biểu hiện", "dấu hiệu" nếu phía sau nó là một triệu chứng độc lập (VD: chỉ bọc <trieu_chung>đánh trống ngực</trieu_chung> thay vì "cảm giác đánh trống ngực"). CHỈ GIỮ LẠI từ "cảm giác" nếu theo sau là tính từ (VD: <trieu_chung>cảm giác kiến bò</trieu_chung>).
- TỪ CẤM (Phải loại ra khỏi thẻ): KHÔNG được đưa các từ chỉ hành động, diễn tiến vào trong thẻ (VD: "chẩn đoán", "phát hiện", "ngừng", "trở nên", "tồi tệ hơn").
- Không gộp: Không gộp 2 thực thể độc lập đứng cạnh nhau. Phải bọc thẻ riêng biệt.
- Không sửa lỗi chính tả: GIỮ NGUYÊN VĂN BẢN GỐC, KHÔNG ĐƯỢC PHÓNG TÁC, KHÔNG ĐƯỢC THÊM HAY BỚT CHỮ CỦA VĂN BẢN. Chỉ được thêm thẻ <tag></tag>.
- Output: CHỈ in ra văn bản đã bọc thẻ XML, tuyệt đối KHÔNG giải thích, KHÔNG thêm Markdown (```xml).

[VĂN BẢN ĐẦU VÀO]
{text}
"""

TAG_MAPPING = {
    "chan_doan": "CHẨN_ĐOÁN",
    "thuoc": "THUỐC",
    "trieu_chung": "TRIỆU_CHỨNG",
    "ten_xet_nghiem": "TÊN_XÉT_NGHIỆM",
    "ket_qua_xet_nghiem": "KẾT_QUẢ_XÉT_NGHIỆM"
}

def parse_tagged_text(tagged_text):
    # Loại bỏ thẻ code markdown nếu Gemini vô tình trả về
    tagged_text = tagged_text.replace("```xml\n", "").replace("```html\n", "").replace("```\n", "").replace("```", "")
    
    pattern = re.compile(r'<([a-z_]+)>([^<]+)</\1>')
    clean_text = ""
    entities = []
    last_end = 0
    
    for match in pattern.finditer(tagged_text):
        tag_name = match.group(1)
        entity_text = match.group(2)
        
        clean_text += tagged_text[last_end:match.start()]
        start_pos = len(clean_text)
        clean_text += entity_text
        end_pos = len(clean_text)
        
        if tag_name in TAG_MAPPING:
            entities.append({
                "text": entity_text,
                "type": TAG_MAPPING[tag_name],
                "position": [start_pos, end_pos]
            })
            
        last_end = match.end()
        
    clean_text += tagged_text[last_end:]
    return clean_text, entities

def process_file(txt_path, gt_dir):
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    
    if not raw_text: return
    
    prompt = PROMPT_TEMPLATE.format(text=raw_text)
    
    print(f"Đang gửi {os.path.basename(txt_path)} tới Gemini...")
    try:
        response = model.generate_content(prompt)
        tagged_text = response.text
    except Exception as e:
        print(f"❌ Lỗi khi gọi API cho {txt_path}: {e}")
        return
        
    # Xử lý kết quả để tách text sạch và tính tọa độ json
    clean_text, entities = parse_tagged_text(tagged_text)
    
    basename = os.path.basename(txt_path)
    name_no_ext = os.path.splitext(basename)[0]
    out_json_path = os.path.join(gt_dir, f"{name_no_ext}.json")
    
    # ⚠️ QUAN TRỌNG: Ghi đè lại file text gốc bằng clean_text từ LLM!
    # Vì đôi khi LLM lén tự sửa lỗi chính tả hoặc thừa thiếu khoảng trắng. 
    # Nếu không ghi đè lại, tọa độ (offset) sẽ bị trệch so với file text gốc.
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
        
    # Lưu file json chứa entity
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=4)
        
    print(f"  -> Thành công: Đã lưu {len(entities)} entities vào {name_no_ext}.json")

def main():
    text_dir = r"D:\SideProject\viettel_ai_race\code\data\input\input"
    gt_dir = r"D:\SideProject\viettel_ai_race\code\data\output\output"
    
    # Đảm bảo thư mục gt đã tồn tại
    os.makedirs(gt_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(text_dir, "*.txt"))
    if not txt_files:
        print(f"Không tìm thấy file .txt nào trong {text_dir}")
        return
        
    for fpath in txt_files:
        basename = os.path.basename(fpath)
        name_no_ext = os.path.splitext(basename)[0]
        json_path = os.path.join(gt_dir, f"{name_no_ext}.json")
        
        # Bỏ qua nếu đã có file JSON tương ứng (tránh gọi lại API tốn tiền)
        if os.path.exists(json_path):
            print(f"Bỏ qua {basename} vì đã có json.")
            continue
            
        process_file(fpath, gt_dir)
        time.sleep(3) # Nghỉ 3s để tránh bị API chặn vì Rate Limit (quá tải)

if __name__ == "__main__":
    main()
