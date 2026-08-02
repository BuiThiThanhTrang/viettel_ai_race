import os
import json
import glob

data_dir = r"D:\SideProject\viettel_ai_race\code\data\finetune2"
text_dir = os.path.join(data_dir, "text")
gt_dir = os.path.join(data_dir, "gt")
output_file = os.path.join(data_dir, "dataset.jsonl")

# Tìm tất cả các file txt
txt_files = glob.glob(os.path.join(text_dir, "*.txt"))

instruction = "Trích xuất các thực thể y tế (ví dụ: Thuốc, Triệu chứng, Bệnh, ...) và thông tin liên quan từ đoạn văn bản dưới đây. Trả về kết quả dưới dạng mảng JSON."

with open(output_file, "w", encoding="utf-8") as f_out:
    for txt_path in txt_files:
        basename = os.path.basename(txt_path)
        filename_no_ext = os.path.splitext(basename)[0]
        json_path = os.path.join(gt_dir, filename_no_ext + ".json")
        
        if os.path.exists(json_path):
            with open(txt_path, "r", encoding="utf-8") as f_txt:
                text_content = f_txt.read().strip()
                
            with open(json_path, "r", encoding="utf-8") as f_json:
                gt_content = json.load(f_json)
                
            # Tạo dictionary chuẩn instruction format
            data_point = {
                "instruction": instruction,
                "input": text_content,
                "output": json.dumps(gt_content, ensure_ascii=False)
            }
            
            # Ghi vào file jsonl
            f_out.write(json.dumps(data_point, ensure_ascii=False) + "\n")

print(f"Đã tạo thành công file {output_file} chứa {len(txt_files)} mẫu dữ liệu!")
