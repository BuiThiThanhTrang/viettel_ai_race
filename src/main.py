import os
import sys
import argparse
from pipeline import MedicalExtractionPipeline
from utils import read_text_file, write_json_file

# Fix lỗi in tiếng Việt ra Windows Console
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Medical Text Extraction End-to-End Pipeline")
    parser.add_argument("--input_dir", type=str, default=r"..\data\input\input", help="Thư mục chứa các file text đầu vào")
    parser.add_argument("--output_dir", type=str, default=r"..\data\output\output", help="Thư mục chứa các file JSON đầu ra")
    parser.add_argument("--use_llm", action="store_true", default=False, help="Sử dụng LLM 7B thay vì Encoder model cho NER")
    parser.add_argument("--use_llm_api", action="store_true", default=True, help="Sử dụng LLM qua API thay vì tải local GPU")
    parser.add_argument("--llm_api_url", type=str, default="http://localhost:8000/v1/chat/completions", help="URL của LLM API")
    parser.add_argument("--api_key", type=str, default=None, help="API Key (Bearer Token) nếu API Server yêu cầu xác thực")
    parser.add_argument("--assertion_mode", type=str, choices=["rule_based", "llm", "encoder"], default="rule_based", help="Chế độ phân loại ngữ cảnh")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(input_dir):
        print(f"Lỗi: Không tìm thấy thư mục {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("Khởi tạo Pipeline...")
    if args.use_llm:
        from pipeline import MedicalNERLLM
        ner_model = MedicalNERLLM(use_api=args.use_llm_api, api_url=args.llm_api_url, api_key=args.api_key)
    else:
        from pipeline import MedicalNEREncoder
        ner_model = MedicalNEREncoder()
        
    pipeline = MedicalExtractionPipeline(ner_model=ner_model, assertion_mode=args.assertion_mode)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    print(f"Bắt đầu xử lý {len(txt_files)} file trong thư mục {input_dir}...")

    for i, filename in enumerate(txt_files):
        input_path = os.path.join(input_dir, filename)
        output_filename = filename.replace(".txt", ".json")
        output_path = os.path.join(output_dir, output_filename)

        print(f"[{i+1}/{len(txt_files)}] Đang xử lý: {filename}...")
        text = read_text_file(input_path)
        
        # Xử lý trích xuất
        results = pipeline.process_text(text)
        
        # Ghi file JSON
        write_json_file(results, output_path)

    print(f"\nHoàn thành! Toàn bộ kết quả đã được lưu tại thư mục: {output_dir}")

if __name__ == "__main__":
    main()
