from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="1d4ab069f67c65a0b083f7b223a880cd03f001e0e942d9f2b363cd38e39f623c",
)

chunk = "triệu chứng: mang thai, mắc ói, đau đầu"

prompt = f"""Bạn là một chuyên gia y tế AI. Trích xuất chính xác 100% nguyên văn các thực thể y khoa từ đoạn văn bản sau và phân thành 5 loại: CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.
Tuyệt đối không tóm tắt hay thay đổi từ ngữ gốc. Chỉ trả về một mảng JSON duy nhất.

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

response = client.chat.completions.create(
    model="ii-medical-8b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a medical language model. "
                "Provide cautious and evidence-oriented answers."
            ),
        },
        {
            "role": "user",
            "content": f"{prompt}",
        },
    ],
    temperature=0.2,
    max_tokens=512,
)

print(response.choices[0].message.content)