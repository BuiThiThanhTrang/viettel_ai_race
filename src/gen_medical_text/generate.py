import os
import time
from google import genai
from google.genai import types

# CẤU HÌNH
OUTPUT_DIR = r"D:\SideProject\viettel_ai_race\code\data\finetune\text"
NUM_SAMPLES = 50
# Lưu ý: Thay YOUR_GEMINI_API_KEY bằng key thật của bạn
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE") 
MODEL_NAME = "gemini-3.5-flash-lite" # Model rất phù hợp để tạo dữ liệu sinh động

example_13 = """Câu hỏi từ người dùng:

Em chào bác sỹ
Cho em hỏi em có bị thương nhẹ ở tay nhưng bị chảy máu,sau đó khoảng 10 phút em có chơi với 1 con chó con nhà em (chó chưa tiêm dại) tuổi có có dính nước dãi của chó con vào tay. Hôm qua em vô tình đọc được thông tin là nếu dính nước dãi chó vẫn có nguy cơ lây bệnh dại. Vậy bác sỹ cho em hỏi trong trường hợp của em có bị sao không ạ?
Mong nhận được sự tư vấn của bác sỹ
Câu trả lời của bác sĩ:

Chào bạn,
Để giải đáp thắc mắc của bạn, tôi xin chia sẻ một số thông tin như sau:
Bệnh dại chỉ không chỉ lây qua vết cắn của động vật. Đường lây bệnh dại phổ biến nhất là do bị động vật dại cắn. Bệnh dại còn có thể lây truyền từ nước bọt của chó, mèo dại hoặc động vật khác mắc bệnh dại do cào hoặc liếm vào vết thương, những vùng da bị trầy xước của cơ thể.
1. Bệnh dại có lây không?
Bệnh dại do Lyssavirus, thuộc họ Lyssaviridae gây ra. Sau khi xâm nhập vào cơ thể người và động vật có vú, virus di chuyển theo hệ thần kinh vào tủy sống và não, phá hủy các trung khu thần kinh trong đại não, gây ra trạng thái điên dạiở động vật và người.
Bệnh dại là bệnh đe dọa tính mạng và có thể gây tử vong cho người nếu người bị cắn không rửa vết thương và được điều trị y tế kịp thời sau khi bị cắn. Không có thuốc điều trị khi lên cơn dại. Phòng bệnh bằng tiêm vaccine phòng dại. Bệnh dạithường gia tăng vào mùa hè.
Bệnh dại gây ra bởi virus, và là bệnh lây truyền. Do vậy, việc phòng bệnh dại là vô cùng cần thiết.
2. Bệnh dại lây truyền qua đường nào?
Vi-rút dại chủ yếu được lây truyền từ nước bọt của các loài động vật bị dại sang người qua vết cắn hoặc qua vết trầy xước trên cơ thể con người.
- 96% các trường hợp gây bệnh dại ở người tại Đông Nam Á là do chó cắn. Nơi bị chó cắn càng gần thần kinh trung ương thì nạn nhân càng phát bệnh nhanh.
- Thế giới ghi nhận việc lây bệnh qua không khí có thể xảy ra khi ở trong hang dơi hay tiếp xúc với chất thải của dơi, việc lây bệnh qua không khí này đã được ghi nhận tại 4 báo cáo về ca mắc bệnh dại ở người và liên quan tới công việc thí nghiệm với động vật. Tuy nhiên các ca mắc dạng này chưa ghi nhận tại Việt Nam.
- Các yếu tố có thể ảnh hưởng đến sự phát triển lây nhiễm bệnh dạibao gồm:
2.  Tiền sử bệnh hiện tại
    Lý do nhập viện: đau ngực trái cấp tính và đau sau xương ức lan ra sau lưng
Mức độ nghiêm trọng của vết cắn
Số lượng vi rút dại xâm nhập vào
Tình trạng miễn dịch của bệnh nhân
Vùng bị cắn - vết thương ở đầu và cổ, cũng như những vết thương ở các khu vực đầu mút thần kinh như ngón tay, thường có thời gian ủ bệnh ngắn hơn do khoảng cách gần hơn cho vi rút xâm nhập vào mô thần kinh.
Trường hợp của bạn nên đi đến các trung tâm tiêm chủng gần nhất để kiểm tra y tế và chích ngừa bạn nhé!
"""

example_37 = """Hỏi : Bé nhà mình là bé trai, khi bé đc 20 ngày tuổi mình mới phát hiện tai phải bé không có lỗ tai, vẫn có vành tai đầy đủ, mình có đưa bé đi khám ở 1 vài nơi nhưng vì bé còn quá nhỏ nên bs chỉ nhìn bên ngoài và đo âm tai còn lại. Bé nay đã 2 tuổi, nghe nói bình thường, mọi sinh hoạt bình thường, phát triển đều, nhưng mình cảm thấy thật sự lo lắng vì bên tai ko có lỗ kia vành tai không phát triển đều, nó nhỏ hơn tai còn lại. Ngược lại, bên tai trái thì rất to và vành tai đẹp. Mình cũng để ý con xem có phản ứng khi mình gọi không và bé vẫn nghe bìng thường, kể cả khi sắp ngủ mình nói nhỏ bé vẫn nghe và mở mắt nhìn. Mình rất lo lắng về tai của bé, sợ sau này sẽ ảnh hưởng đến sự phát triển của bé và bé bị trêu chọc. Y học hiện nay đã có nghiên cứu gì về trường hợp như thế này chưa nhủ các bạn. Mình có cách gì kiểm tra việc nghe của bé thường xuyên để theo dõi không? Mọi người cho mình xin ý kiến nhé!
    - Vị trí: âm hộ bên phải, mông bên phải
    - Mức độ nghiêm trọng: cực kỳ đau đớn
    - Thời gian: Tình trạng ngày càng nặng trong 5 ngày
    - Các triệu chứng liên quan: ban đỏ, chảy mủ
    Các sự kiện trước khi nhập viện
    - Được bác sĩ chăm sóc chính thay thế khám vào ngày, được tăng liều bactrim và doxycycline để điều trị chẩn đoán Viêm mô tế bào
    - Báo cáo có cải thiện một phần về ban đỏ
    - dịch tiết có vẻ như mủ từ một số tổn thương vào ngày, nhưng tình trạng này đã tự khỏi
    - Khám tại phòng khám vào ngày nhập viện
Dù hiện tại bé nghe khá tốt, nhưng việc chỉ nghe một bên có thể làm con lúng túng khi xác định hướng âm thanh hoặc ở môi trường ồn ào. Để theo dõi tại nhà, bạn hãy thử gọi con từ nhiều góc khuất phía bên phải xem bé có nhanh chóng quay đúng hướng không, và chú ý xem con phát âm có tròn vành rõ chữ không.

Bé hai tuổi đã có thể đo thính lực rất chính xác. Bạn nên đưa con đến các bệnh viện lớn có chuyên khoa Tai mũi họng nhi để đánh giá lại toàn diện. Gia đình hãy cứ lạc quan chăm sóc bé như bình thường, lộ trình điều trị phía trước rất rõ ràng và hoàn toàn khả thi.
"""

SYSTEM_PROMPT = f"""Bạn là một chuyên gia tạo dữ liệu y khoa nhằm mục đích kiểm thử mô hình AI (Adversarial Data Generation).
Nhiệm vụ của bạn là sinh ra các đoạn văn bản y khoa cố tình trộn lẫn giữa hai loại văn phong:
1. Hỏi đáp (Q&A) trên diễn đàn sức khỏe (bệnh nhân hỏi, bác sĩ trả lời về kiến thức chung chung).
2. Hồ sơ bệnh án lâm sàng đặc thù của người bệnh (tiền sử, triệu chứng, chẩn đoán, thuốc kê đơn) BỊ CHÈN VÀO GIỮA đoạn Q&A một cách đột ngột.

YÊU CẦU QUAN TRỌNG:
1. Mỗi lần sinh, hãy tạo ra 1 MẪU văn bản duy nhất. Không đánh số mẫu, không giải thích. Chỉ in ra nội dung.
2. Đa dạng hóa các loại bệnh, đa dạng hóa triệu chứng, tên thuốc, tên xét nghiệm.
3. Độ dài mỗi mẫu từ 150 đến 400 từ.
4. Bố cục CHẮC CHẮN PHẢI CHỨA SỰ ĐỨT GÃY: Đang viết kiểu câu hỏi của người bệnh hoặc kiến thức chung của bác sĩ, tự nhiên chèn vào một đoạn giống như copy từ hồ sơ bệnh viện (Ví dụ: "Tiền sử bệnh hiện tại...", "Lý do nhập viện...", "Khám thực thể: ...", "Chẩn đoán: ...", "Điều trị bằng: ...").
5. KHÔNG dùng markdown định dạng phức tạp (như in đậm, in nghiêng).
6. TUYỆT ĐỐI KHÔNG sinh ra các tình huống liên quan đến bệnh dạ dày, viêm loét dạ dày, trào ngược dạ dày hay các triệu chứng đau dạ dày (đau thượng vị, buồn nôn...). Hãy ưu tiên các bệnh lý khác như: tim mạch, thần kinh, cơ xương khớp, hô hấp, nội tiết, ung bướu...
7. MỤC TIÊU CỐT LÕI - TRIỆU CHỨNG PHỦ ĐỊNH: BẮT BUỘC trong mỗi mẫu phải có ít nhất 1-2 câu mô tả sự vắng mặt của triệu chứng bằng các từ phủ định (Ví dụ: "Bệnh nhân không ho, không sốt", "Người bệnh phủ nhận việc đau ngực", "Chưa ghi nhận dấu hiệu khó thở", "Không yếu liệt"). 
8. MỤC TIÊU CỐT LÕI - TRIỆU CHỨNG PHỨC TẠP: Cố ý chèn các cụm mô tả triệu chứng dài, mang tính hành vi hoặc vận động (Ví dụ: "không thể tự đứng dậy do yếu sức chân phải", "liên tục khuỵu chân khi đi bộ", "mất thăng bằng và ngã sang trái", "cảm giác yếu sức nửa người").
Dưới đây là 2 mẫu ví dụ hoàn hảo về sự trộn lẫn kỳ quặc này để bạn bắt chước phong cách (hãy tạo ra nội dung mới, đừng copy y hệt):

MẪU VÍ DỤ 1:
{example_13}

MẪU VÍ DỤ 2:
{example_37}
"""

USER_PROMPT = "Hãy sinh ra một mẫu văn bản bị trộn lẫn (giữa Q&A và bệnh án lâm sàng) mới về một tình trạng bệnh ngẫu nhiên, không trùng lặp với các mẫu trước đó."

def main():
    if API_KEY == "YOUR_GEMINI_API_KEY":
        print("Vui lòng thay thế YOUR_GEMINI_API_KEY bằng API key thật của bạn (mở file này lên và dán key vào nhé).")
        return

    print("Đang khởi tạo Gemini Client...")
    client = genai.Client(api_key=API_KEY)
    
    # Tạo thư mục nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Đếm số lượng file đã có để tiếp tục (resume) nếu quá trình bị gián đoạn
    existing_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')])
    start_index = existing_files + 1
    
    # Nếu trong thư mục đã có quá 50 file (ví dụ: đã có 1000 file như cấu hình trước)
    # thì ta sẽ sinh ra các file có tên qa_1.txt, qa_2.txt... để không bị ghi đè.
    print(f"Sẽ sinh {NUM_SAMPLES} mẫu Q&A...")

    for i in range(1, NUM_SAMPLES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=USER_PROMPT,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=1.5, # Nhiệt độ cao để sinh ra đa dạng
                )
            )
            
            text = response.text.strip()
            
            # Lưu file với định dạng qa_<index>.txt để phân biệt với các mẫu thường
            file_path = os.path.join(OUTPUT_DIR, f"{400+i}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            print(f"[{i}/{NUM_SAMPLES}] Đã lưu mẫu bệnh án Q&A trộn lẫn vào {file_path}")
            
            # Nghỉ một chút để tránh dính giới hạn API (Rate Limit) nếu dùng tài khoản miễn phí
            time.sleep(2)
            
        except Exception as e:
            print(f"Lỗi khi sinh mẫu thứ {i}: {e}")
            print("Đợi 10 giây trước khi thử lại...")
            time.sleep(10)
            
if __name__ == "__main__":
    main()
