import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Đường dẫn thư mục data gốc
DATA_DIR = r"d:\SideProject\viettel_ai_race\code\data\db"
ICD10_EXCEL_PATH = os.path.join(DATA_DIR, "DM ICD10-19_8_BYT.xlsx")
RXNORM_RRF_PATH = os.path.join(DATA_DIR, r"RxNorm_full_prescribe_07062026\rrf\RXNCONSO.RRF")

# Đường dẫn file output
OUTPUT_DIR = r"D:\SideProject\viettel_ai_race\code\data\clean"
ICD10_OUT_PATH = os.path.join(OUTPUT_DIR, "icd10_clean.json")
RXNORM_OUT_PATH = os.path.join(OUTPUT_DIR, "rxnorm_clean.json")

def process_icd10():
    print(f"Đang xử lý ICD-10 từ file: {ICD10_EXCEL_PATH}")
    # Bắt đầu từ hàng 5 -> skiprows=4
    # Cột B, C, D -> usecols="B:D"
    df = pd.read_excel(ICD10_EXCEL_PATH, skiprows=4, usecols="B:D", names=["Mã bệnh", "Tên bệnh", "Nhóm bệnh"])
    
    # Loại bỏ các hàng bị rỗng mã bệnh
    df = df.dropna(subset=["Mã bệnh", "Tên bệnh"])
    
    icd_dict = {}
    for _, row in df.iterrows():
        code = str(row["Mã bệnh"]).strip()
        name = str(row["Tên bệnh"]).strip()
        if code and name:
            icd_dict[code] = name
            
    print(f"Đã trích xuất {len(icd_dict)} mã ICD-10.")
    return icd_dict

def process_rxnorm():
    print(f"Đang xử lý RxNorm từ file: {RXNORM_RRF_PATH}")
    rx_dict = {}
    
    # RRF là file phân cách bằng dấu pipe |
    # Cột 0: RxCUI (Mã chuẩn)
    # Cột 1: LAT (Ngôn ngữ, thường chọn ENG)
    # Cột 14: STR (String name)
    
    count = 0
    with open(RXNORM_RRF_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) > 14:
                rxcui = parts[0].strip()
                lat = parts[1].strip()
                name = parts[14].strip()
                
                if lat == 'ENG' and rxcui and name:
                    # RxNorm có thể có nhiều tên chéo nhau cho 1 mã, ta lấy tên cuối cùng lưu lại để giảm size
                    rx_dict[rxcui] = name
            count += 1
            if count % 50000 == 0:
                print(f"  Đã đọc {count} dòng...")

    print(f"Đã trích xuất {len(rx_dict)} mã RxNorm (unique).")
    return rx_dict

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. ICD-10
    if os.path.exists(ICD10_EXCEL_PATH):
        icd_data = process_icd10()
        with open(ICD10_OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(icd_data, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu ICD-10 ra: {ICD10_OUT_PATH}\n")
    else:
        print(f"Không tìm thấy file ICD-10 tại {ICD10_EXCEL_PATH}")

    # 2. RxNorm
    if os.path.exists(RXNORM_RRF_PATH):
        rx_data = process_rxnorm()
        with open(RXNORM_OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(rx_data, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu RxNorm ra: {RXNORM_OUT_PATH}\n")
    else:
        print(f"Không tìm thấy file RxNorm tại {RXNORM_RRF_PATH}")

if __name__ == "__main__":
    main()
