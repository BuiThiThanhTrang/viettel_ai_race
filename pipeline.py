import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import utils
import json
import os
import requests

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from underthesea import word_tokenize

class MedicalNERBase:
    def extract(self, text):
        raise NotImplementedError("Subclasses must implement this method")

class MedicalNEREncoder(MedicalNERBase):
    def __init__(self, model_name="cbc-528a/BamiBERT-ViMedNER"):
        print(f"Đang tải tokenizer và mô hình NER từ {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        print(f"Đã tải xong mô hình {model_name}!")
        
        # BamiBERT trả về: ten_benh, trieu_chung_benh, bien_phap_dieu_tri...
        self.label_mapping = {
            "ten_benh": "CHẨN_ĐOÁN",
            "trieu_chung_benh": "TRIỆU_CHỨNG",
            "bien_phap_dieu_tri": "THUỐC",
            "bien_phap_chan_doan": "TÊN_XÉT_NGHIỆM",
            "nguyen_nhan_benh": "CHẨN_ĐOÁN"
        }

    def extract(self, text):
        entities = []
        
        # Tách từ bằng underthesea (pure Python, không cần Java)
        # format="text" sẽ trả về chuỗi với các từ ghép được nối bằng gạch dưới _
        processed_text = word_tokenize(text, format="text")
  
        # Xử lý theo từng dòng
        lines = processed_text.split('\n')
        current_offset = 0
        
        for line in lines:
            if not line.strip():
                current_offset += len(line) + 1 # +1 cho \n
                continue
            
            # Chia dòng thành nhiều chunks nhỏ hơn 400 ký tự để không bị mất dữ liệu
            max_len = 400
            words = line.split(' ')
            chunks = []
            chunk_starts = [] # start offset của từng chunk so với đầu dòng
            
            curr_chunk = []
            curr_len = 0
            curr_start = 0
            
            for w in words:
                if curr_len + len(w) > max_len and curr_chunk:
                    chunk_text = ' '.join(curr_chunk)
                    chunks.append(chunk_text)
                    chunk_starts.append(curr_start)
                    
                    curr_start += len(chunk_text) + 1 # +1 cho dấu cách
                    curr_chunk = [w]
                    curr_len = len(w) + 1
                else:
                    curr_chunk.append(w)
                    curr_len += len(w) + 1
            if curr_chunk:
                chunks.append(' '.join(curr_chunk))
                chunk_starts.append(curr_start)
                
            for chunk, start_in_line in zip(chunks, chunk_starts):
                hf_entities = self.nlp(chunk)
            
                for ent in hf_entities:
                    word = ent.get('word', '').replace('@@', '').replace('_', ' ').strip()
                    if not word: continue
                    
                    raw_label = ent.get('entity_group', ent.get('entity', ''))
                    # Lấy nhãn cốt lõi (bỏ B-, I-)
                    core_label = raw_label.replace('B-', '').replace('I-', '')
                    
                    mapped_type = self.label_mapping.get(core_label, core_label)
                    # Chỉ lấy các type hợp lệ
                    if mapped_type not in ["CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"]:
                        continue
                    
                    start_val = ent.get('start')
                    end_val = ent.get('end')
                    # Tọa độ tuyệt đối = đầu dòng + đầu chunk + vị trí trong chunk
                    start_pos = current_offset + start_in_line + (start_val if start_val is not None else 0)
                    end_pos = current_offset + start_in_line + (end_val if end_val is not None else 0)
                    
                    # Tìm lại từ gốc (chưa có dấu _) trong văn bản gốc
                    exact_word, pos = utils.find_exact_position(text, word, start_pos)
                    if not pos:
                        pos = [start_pos, end_pos]
                        exact_word = word
                    
                    entities.append({
                        "text": exact_word,
                        "type": mapped_type,
                        "position": pos
                    })
            
            current_offset += len(line) + 1
        
        entities = sorted(entities, key=lambda x: x["position"][0] if x["position"] else 0)
        return entities

class MedicalNERLLM(MedicalNERBase):
    def __init__(self, use_api=True, api_url="http://localhost:8000/v1/chat/completions", model_name="Intelligent-Internet/II-Medical-8B", api_key=None):
        self.use_api = use_api
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        
        if not self.use_api:
            print(f"Đang tải LLM {model_name} vào GPU qua transformers...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            print("Đã tải xong LLM vào GPU!")
        else:
            print(f"Đã cấu hình gọi LLM qua API: {api_url}")

    def extract(self, text):
        # 1. Áp dụng Chunking: Chia nhỏ đoạn văn bản thành các chunks để tránh bị cắt JSON
        max_len = 400
        words = text.split(' ')
        chunks = []
        chunk_starts = []
        
        curr_chunk = []
        curr_len = 0
        curr_start = 0
        
        for w in words:
            if curr_len + len(w) > max_len and curr_chunk:
                chunk_text = ' '.join(curr_chunk)
                chunks.append(chunk_text)
                chunk_starts.append(curr_start)
                
                curr_start += len(chunk_text) + 1
                curr_chunk = [w]
                curr_len = len(w) + 1
            else:
                curr_chunk.append(w)
                curr_len += len(w) + 1
        if curr_chunk:
            chunks.append(' '.join(curr_chunk))
            chunk_starts.append(curr_start)
            
        all_entities = []
        
        for chunk, start_in_text in zip(chunks, chunk_starts):
            # 2. Few-shot Prompting: Ép LLM trích xuất chính xác 100% nguyên văn
            system_prompt = "Bạn là một API Server y khoa. Nhiệm vụ của bạn là trích xuất dữ liệu. BẠN CHỈ ĐƯỢC PHÉP TRẢ VỀ JSON ARRAY. Tuyệt đối không được giải thích, không được chào hỏi, không được dùng markdown (```json). Chỉ xuất ra đúng một mảng chứa ngoặc vuông."
            prompt = f"""Trích xuất chính xác 100% nguyên văn các thực thể y khoa từ đoạn văn bản sau và phân thành 5 loại: CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.

Ví dụ:
Văn bản: "Bệnh nhân ho nhiều, sốt cao. Bác sĩ chỉ định chụp X-quang phổi kết quả bình thường. Kê đơn paracetamol 500mg."
Kết quả JSON:
[
  {{"text": "ho nhiều", "type": "TRIỆU_CHỨNG"}},
  {{"text": "sốt cao", "type": "TRIỆU_CHỨNG"}},
  {{"text": "chụp X-quang phổi", "type": "TÊN_XÉT_NGHIỆM"}},
  {{"text": "bình thường", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}},
  {{"text": "paracetamol 500mg", "type": "THUỐC"}}
]

Văn bản: "{chunk}"
Kết quả JSON:"""

                    payload = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.0, # Bắt buộc dùng 0.0 để loại bỏ hoàn toàn tính ngẫu nhiên/ảo giác
                        "max_tokens": 2048 # Tăng giới hạn token vì các mô hình reasoning rất tốn token để "suy nghĩ"
                    }
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                        
                    response = requests.post(self.api_url, json=payload, headers=headers)
                    if response.status_code == 200:
                        content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        # Tiền xử lý: Cắt bỏ hoàn toàn quá trình "suy nghĩ" của các mô hình Reasoning (DeepSeek-R1, Qwen...)
                        import re
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        # Nếu output bị ngắt giữa chừng và chưa có thẻ đóng </think>
                        if '<think>' in content:
                            content = content.split('</think>')[-1]
                            
                        chunk_entities = self._parse_json(content, chunk, start_in_text)
                        all_entities.extend(chunk_entities)
                else:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.0)
                    content = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    chunk_entities = self._parse_json(content, chunk, start_in_text)
                    all_entities.extend(chunk_entities)
            except Exception as e:
                print(f"Lỗi khi gọi LLM cho chunk: {e}")
                
        # Sắp xếp lại theo vị trí xuất hiện
        all_entities = sorted(all_entities, key=lambda x: x["position"][0] if x["position"] else 0)
        return all_entities
        
    def _parse_json(self, content, chunk_text, start_in_text):
        entities = []
        import re
        try:
            # 3. Trích xuất JSON bằng Regex để bỏ bọc Markdown (Dirty JSON Parsing)
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if not match:
                # 4. Thuật toán cứu hộ JSON (Rescue truncated JSON)
                content = content.strip()
                if content.startswith('['):
                    if not content.endswith(']'):
                        if content.endswith('}'):
                            content += ']'
                        elif content.endswith('"'):
                            content += '}]'
                        else:
                            content += '"}]'
                    match = re.search(r'\[.*\]', content, re.DOTALL)

            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                
                search_index = 0
                for item in parsed:
                    word = item.get("text", "").strip()
                    ent_type = item.get("type", "")
                    
                    # Bộ lọc Type (Chặn LLM ảo giác chế ra Type mới)
                    if not word or ent_type not in ["CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"]:
                        continue
                        
                    # Tìm tọa độ nội bộ trong phạm vi chunk
                    exact_word, local_pos = utils.find_exact_position(chunk_text, word, search_index)
                    if not local_pos:
                        # Fallback: dò lại từ đầu chunk đề phòng LLM bỏ sót từ trước đó
                        exact_word, local_pos = utils.find_exact_position(chunk_text, word, 0)
                        
                    if local_pos:
                        # Quy đổi ra tọa độ tuyệt đối của toàn bộ file text
                        pos = [start_in_text + local_pos[0], start_in_text + local_pos[1]]
                        entities.append({
                            "text": exact_word,
                            "type": ent_type,
                            "position": pos
                        })
                        search_index = local_pos[1]
        except Exception as e:
            print(f"Lỗi parse JSON từ LLM: {e}")
        return entities

class AssertionClassifierBase:
    def classify(self, entity, text_context, window_size=50):
        raise NotImplementedError()

class AssertionClassifierRuleBased(AssertionClassifierBase):
    def __init__(self):
        # Tập quy tắc rule-based đơn giản để bắt ngữ cảnh
        self.negation_keywords = ["không có", "không bị", "phủ nhận", "chưa phát hiện"]
        self.family_keywords = ["gia đình", "bố", "mẹ", "anh", "chị", "em", "ông", "bà"]
        self.history_keywords = ["tiền sử", "trước đây", "đã điều trị", "cũ", "năm ngoái", "trước khi nhập viện"]

    def classify(self, entity, text_context, window_size=50):
        ent_type = entity.get("type", "")
        if ent_type not in ["CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"]:
            return []

        assertions = []
        pos = entity.get("position", [0, 0])
        
        # Cắt đoạn ngữ cảnh xung quanh thực thể
        start_ctx = max(0, pos[0] - window_size)
        end_ctx = min(len(text_context), pos[1] + window_size)
        context = text_context[start_ctx:end_ctx].lower()

        # Kiểm tra Negation
        for kw in self.negation_keywords:
            if kw in context:
                assertions.append("isNegated")
                break
                
        # Kiểm tra Family
        for kw in self.family_keywords:
            if kw in context:
                assertions.append("isFamily")
                break
                
        # Kiểm tra Historical
        for kw in self.history_keywords:
            if kw in context:
                assertions.append("isHistorical")
                break

        return assertions

class AssertionClassifierLLM(AssertionClassifierBase):
    def __init__(self, api_url="http://localhost:8000/v1/chat/completions", model_name="II-Vietnam/II-Medical-8B-SFT"):
        self.api_url = api_url
        self.model_name = model_name

    def classify(self, entity, text_context, window_size=50):
        ent_text = entity.get("text", "")
        pos = entity.get("position", [0, 0])
        
        start_ctx = max(0, pos[0] - window_size * 2)
        end_ctx = min(len(text_context), pos[1] + window_size * 2)
        context = text_context[start_ctx:end_ctx]
        
        prompt = f"""Đọc đoạn văn bản sau: "{context}".
Thực thể "{ent_text}" trong đoạn văn này có ý nghĩa phủ định (isNegated), tiền sử gia đình (isFamily), hay tiền sử bệnh án cũ (isHistorical) không? 
Trả lời bằng JSON array chứa các nhãn hợp lệ trong số ["isNegated", "isFamily", "isHistorical"]. Nếu không có nhãn nào phù hợp, hãy trả về mảng rỗng []. Không giải thích thêm.
Kết quả JSON:"""

        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            response = requests.post(self.api_url, json=payload, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx != -1 and end_idx != -1:
                    parsed = json.loads(content[start_idx:end_idx])
                    if isinstance(parsed, list):
                        return [p for p in parsed if p in ["isNegated", "isFamily", "isHistorical"]]
        except Exception as e:
            print(f"Lỗi Assertion LLM: {e}")
        return []

class AssertionClassifierEncoder(AssertionClassifierBase):
    def __init__(self, model_path="model_assertion_output"):
        # Yêu cầu phải chạy file train_assertion.py để có model_path này
        if os.path.exists(model_path):
            from transformers import pipeline as hf_pipeline
            self.classifier = hf_pipeline("text-classification", model=model_path, top_k=None)
        else:
            print(f"Cảnh báo: Không tìm thấy model tại {model_path}. Fallback về Rule-based.")
            self.classifier = None
            self.fallback = AssertionClassifierRuleBased()

    def classify(self, entity, text_context, window_size=50):
        if self.classifier is None:
            return self.fallback.classify(entity, text_context, window_size)
            
        pos = entity.get("position", [0, 0])
        
        start_ctx = max(0, pos[0] - window_size)
        end_ctx = min(len(text_context), pos[1] + window_size)
        
        prefix = text_context[start_ctx:pos[0]]
        ent_text = text_context[pos[0]:pos[1]]
        suffix = text_context[pos[1]:end_ctx]
        
        # Đóng gói thẻ [E] và [/E]
        marked_sentence = f"{prefix} [E] {ent_text} [/E] {suffix}"
        
        try:
            results = self.classifier(marked_sentence)[0]
            assertions = []
            for res in results:
                if res['score'] >= 0.5 and res['label'] != "None":
                    assertions.append(res['label'])
            return assertions
        except Exception as e:
            print(f"Lỗi phân loại Encoder Assertion: {e}")
            return []

class CandidateLinker:
    def __init__(self, model_name="demdecuong/vihealthbert-base-word"):
        print(f"Đang tải SentenceTransformer bằng model y khoa {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        # Sửa lỗi: vihealthbert-base-word (dựa trên kiến trúc RoBERTa) có max_position_embeddings = 258.
        # Nếu không giới hạn max_seq_length, SentenceTransformer sẽ cố nạp chuỗi dài hơn dẫn đến lỗi out of bounds.
        self.encoder.max_seq_length = 256
        
        # Load CSDL JSON thật
        icd_docs, self.icd_codes = [], []
        rx_docs, self.rx_codes = [], []
        
        icd_path = r"d:\SideProject\viettel_ai_race\code\data\clean\icd10_clean.json"
        rxnorm_path = r"d:\SideProject\viettel_ai_race\code\data\clean\rxnorm_clean.json"
        
        # Đường dẫn lưu cache embeddings
        icd_emb_path = r"d:\SideProject\viettel_ai_race\code\data\clean\icd10_embeddings.pt"
        rxnorm_emb_path = r"d:\SideProject\viettel_ai_race\code\data\clean\rxnorm_embeddings.pt"
        
        if os.path.exists(icd_path):
            with open(icd_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for code, text in data.items():
                    self.icd_codes.append(code)
                    icd_docs.append(text)
        
        if os.path.exists(rxnorm_path):
            with open(rxnorm_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for code, text in data.items():
                    self.rx_codes.append(code)
                    rx_docs.append(text)
                    
        # Chuyển đổi thành Vector Embeddings có hỗ trợ Caching
        if icd_docs:
            if os.path.exists(icd_emb_path):
                print("Đang nạp bộ Vector Embeddings của ICD-10 từ bộ nhớ đệm (Cache)...")
                self.icd_matrix = torch.load(icd_emb_path, map_location=self.encoder.device, weights_only=True)
            else:
                print("Đang mã hóa CSDL ICD-10 thành Vector Embeddings (lần đầu)...")
                self.icd_matrix = self.encoder.encode(icd_docs, convert_to_tensor=True, show_progress_bar=True)
                torch.save(self.icd_matrix, icd_emb_path)
        
        if rx_docs:
            if os.path.exists(rxnorm_emb_path):
                print("Đang nạp bộ Vector Embeddings của RxNorm từ bộ nhớ đệm (Cache)...")
                self.rx_matrix = torch.load(rxnorm_emb_path, map_location=self.encoder.device, weights_only=True)
            else:
                print("Đang mã hóa CSDL RxNorm thành Vector Embeddings (lần đầu)...")
                self.rx_matrix = self.encoder.encode(rx_docs, convert_to_tensor=True, show_progress_bar=True)
                torch.save(self.rx_matrix, rxnorm_emb_path)
        
        print(f"Đã nạp {len(icd_docs)} mã ICD-10 và {len(rx_docs)} mã RxNorm vào hệ thống Semantic Search.")

    def link(self, entity_text, entity_type):
        """Ánh xạ text sang mã chuẩn dựa vào loại thực thể."""
        if entity_type == "CHẨN_ĐOÁN" and hasattr(self, 'icd_matrix'):
            return self._search_best_match(entity_text, self.icd_matrix, self.icd_codes, threshold=0.75)
        elif entity_type == "THUỐC" and hasattr(self, 'rx_matrix'):
            return self._search_best_match(entity_text, self.rx_matrix, self.rx_codes, threshold=0.75)
        return []

    def _search_best_match(self, query, db_matrix, codes_list, threshold=0.75):
        """Tìm kiếm chuỗi giống nhất bằng Cosine Similarity trên Semantic Embeddings."""
        query_vec = self.encoder.encode([query], convert_to_tensor=True)
        similarities = cos_sim(query_vec, db_matrix)[0] # Trả về tensor 1D
        
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[best_idx].item()
        
        if best_score >= threshold:
            return [codes_list[best_idx]]
        return []

class MedicalExtractionPipeline:
    def __init__(self, ner_model=None, assertion_mode="rule_based"):
        if ner_model is None:
            # Mặc định sử dụng Encoder model (BamiBERT)
            self.ner = MedicalNEREncoder()
        else:
            self.ner = ner_model
            
        if assertion_mode == "llm":
            self.assertion_classifier = AssertionClassifierLLM()
        elif assertion_mode == "encoder":
            self.assertion_classifier = AssertionClassifierEncoder()
        else:
            self.assertion_classifier = AssertionClassifierRuleBased()
            
        self.linker = CandidateLinker()

    def process_text(self, text):
        """Xử lý end-to-end một đoạn văn bản y khoa."""
        # 1. Trích xuất thực thể
        entities = self.ner.extract(text)
        
        # 2. Xử lý từng thực thể: Phân loại Assertion và Ánh xạ mã
        for ent in entities:
            # Phân loại Assertion
            if ent["type"] in ["CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"]:
                ent["assertions"] = self.assertion_classifier.classify(ent, text)
                
            # Ánh xạ Candidate (nếu là Bệnh hoặc Thuốc)
            if ent["type"] in ["CHẨN_ĐOÁN", "THUỐC"]:
                candidates = self.linker.link(ent["text"], ent["type"])
                if candidates:
                    ent["candidates"] = candidates

        return entities
