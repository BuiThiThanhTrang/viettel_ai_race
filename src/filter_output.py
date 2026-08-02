import os
import json
import glob

def is_valid_entity(text):
    text = text.strip().lower()
    
    # 1. Lọc mẫu có độ dài 1 ký tự
    if len(text) <= 1:
        return False
        
    # 2. Lọc các cụm chỉ có phụ âm đầu (tr, ch, ph, nh, gi, th...)
    # Kiểm tra xem chuỗi có chứa ít nhất một nguyên âm tiếng Việt không
    # vowels = "aeiouyáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    # has_vowel = any(char in vowels for char in text)
    # if not has_vowel:
    #     return False
        
    # # 3. Lọc danh sách đen (Blacklist) tùy chỉnh
    # blacklist = ["sức", "bên", "phải", "trái", "của", "và", "nhưng", "tuy", "có", "không", "thể"]
    # if text in blacklist:
    #     return False
        
    return True

def process_all_outputs(output_dir):
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    total_removed = 0
    total_files = 0
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                entities = json.load(f)
            except json.JSONDecodeError:
                continue
                
        if not isinstance(entities, list):
            continue
            
        original_count = len(entities)
        
        # Áp dụng bộ lọc
        cleaned_entities = [ent for ent in entities if is_valid_entity(ent.get("text", ""))]
        
        removed_count = original_count - len(cleaned_entities)
        if removed_count > 0:
            total_removed += removed_count
            total_files += 1
            
            # Ghi đè lại file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_entities, f, ensure_ascii=False, indent=4)
                
    print(f"✅ Đã dọn dẹp xong! Xóa {total_removed} thực thể rác từ {total_files} file JSON.")

if __name__ == "__main__":
    # Đường dẫn trỏ tới thư mục output
    output_dir = r"D:\SideProject\viettel_ai_race\code\data\output\output"
    
    if os.path.exists(output_dir):
        print(f"Đang xử lý các file trong {output_dir}...")
        process_all_outputs(output_dir)
    else:
        print(f"❌ Không tìm thấy thư mục {output_dir}")
