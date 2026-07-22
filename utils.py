import json
import re

def read_text_file(filepath):
    """Đọc nội dung từ file text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_json_file(data, filepath):
    """Ghi dữ liệu JSON ra file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def find_exact_position(original_text, extracted_text, search_start_index=0):
    """
    Tìm vị trí [start, end] chính xác của đoạn text trích xuất trong văn bản gốc.
    Để tối ưu điểm Word Error Rate (WER), cần đảm bảo text trả về lấy chính xác 
    từ nguyên bản của văn bản gốc.
    """
    if not extracted_text:
        return None
        
    # Thoát các ký tự đặc biệt trong Regex
    escaped_text = re.escape(extracted_text.strip())
    
    # Tìm kiếm đoạn text trong văn bản gốc, bắt đầu từ search_start_index
    # để tránh match lại các từ đã match trước đó.
    match = re.search(escaped_text, original_text[search_start_index:], re.IGNORECASE)
    
    if match:
        start_idx = search_start_index + match.start()
        end_idx = search_start_index + match.end()
        # Trả về text gốc (không bị thay đổi in hoa/thường) và vị trí
        return original_text[start_idx:end_idx], [start_idx, end_idx]
    
    return extracted_text, []
