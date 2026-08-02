import os
import json
import glob

def run_sanity_check(text_dir, gt_dir):
    txt_files = glob.glob(os.path.join(text_dir, "*.txt"))
    
    total_files = 0
    total_entities = 0
    error_offset_count = 0
    error_overlap_count = 0
    
    files_with_errors = []

    for txt_path in txt_files:
        basename = os.path.basename(txt_path)
        name_no_ext = os.path.splitext(basename)[0]
        json_path = os.path.join(gt_dir, f"{name_no_ext}.json")
        
        if not os.path.exists(json_path):
            continue
            
        total_files += 1
        
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                entities = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ LỖI CÚ PHÁP JSON: {json_path}")
                continue
            
        file_has_error = False
        
        # Chỉ lấy các entity có trường position hợp lệ và sắp xếp theo vị trí bắt đầu
        valid_entities = [e for e in entities if "position" in e and len(e["position"]) == 2]
        valid_entities.sort(key=lambda x: x["position"][0])
        
        last_end = -1
        
        for idx, ent in enumerate(valid_entities):
            total_entities += 1
            start, end = ent["position"]
            expected_text = ent.get("text", "")
            
            # Kiểm tra 1: Lệch tọa độ (Offset Mismatch)
            # Cắt chuỗi thực tế từ text file bằng tọa độ [start:end]
            actual_text = text[start:end]
            
            if actual_text != expected_text:
                error_offset_count += 1
                file_has_error = True
                print(f"[LOI OFFSET] trong {basename}:")
                print(f"   - Entity goc (JSON) : '{expected_text}'")
                print(f"   - Text bi cat that  : '{actual_text}'")
                print(f"   - Toa do            : [{start}, {end}]")
            
            # Kiểm tra 2: Chồng chéo tọa độ (Overlap)
            if start < last_end:
                error_overlap_count += 1
                file_has_error = True
                print(f"[CANH BAO CHONG CHEO] trong {basename}:")
                print(f"   - '{expected_text}' [{start}, {end}] de len entity truoc do (ket thuc tai {last_end})")
                
            # Cập nhật điểm kết thúc lớn nhất
            if end > last_end:
                last_end = end
                
        if file_has_error:
            files_with_errors.append(basename)
            
    print("\n" + "="*50)
    print("BAO CAO SANITY CHECK TONG QUAN")
    print("="*50)
    print(f"Tong so file da quet  : {total_files}")
    print(f"Tong so entities      : {total_entities}")
    print(f"So loi lech toa do    : {error_offset_count} loi")
    print(f"So loi chong cheo     : {error_overlap_count} loi")
    print(f"So file bi dinh loi   : {len(files_with_errors)} file")
    
    if len(files_with_errors) > 0:
        print("\n-> Ket luan: Du lieu cua ban dang bi loi toa do. Can chay script sua tu dong truoc khi Train!")
    else:
        print("\n-> Tuyet voi! Bo du lieu cua ban hoan hao, san sang cho viec Train!")

if __name__ == "__main__":
    # Sửa đường dẫn nếu cần thiết
    TEXT_DIR = r"D:\SideProject\viettel_ai_race\code\data\fintune\text"
    GT_DIR = r"D:\SideProject\viettel_ai_race\code\data\fintune\gt"
    
    run_sanity_check(TEXT_DIR, GT_DIR)
