import streamlit as st
import google.generativeai as genai
import os
import json
import faiss
import numpy as np

# Configure the API key
genai.configure(api_key="AIzaSyBVBSmE4VSSnxBtr6vsx6wQt700SxMjpsE")

# Đọc và xử lý dữ liệu từ file JSON
def create_combined_text(section):
    section_title = section["section_title"]
    content = section["content"]

    combined_text = section_title + ":"

    for item in content:
        if item["type"] == "p":
            combined_text += " " + item["content"]
        elif item["type"] == "list":
            combined_text += " " + ", ".join(item["items"])
        elif item["type"] == "i":
            combined_text += " " + item["content"]

    return combined_text

# Kiểm tra và tải dữ liệu JSON
with open("presight_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Hàm tạo và lưu embedding hoặc tải từ tệp
def create_and_load_embeddings(data, embeddings_file="embeddings.npy", index_file="faiss_index.index"):
    if os.path.exists(embeddings_file) and os.path.exists(index_file):
        # Tải embeddings và FAISS index từ tệp
        embeddings_matrix = np.load(embeddings_file)
        index = faiss.read_index(index_file)
        print("Đã tải embeddings và index từ tệp.")
    else:
        # Nếu chưa có, tạo mới embeddings và FAISS index
        dimension = 768  # Dimensionality của vector embedding
        index = faiss.IndexFlatL2(dimension)  # FAISS index sử dụng khoảng cách Euclidean (L2)
        embeddings = []  # Lưu trữ embeddings để tham chiếu sau này

        for section in data:
            combined_text = create_combined_text(section)
            result = genai.embed_content(model="models/text-embedding-004", content=combined_text)
            embedding = np.array(result['embedding'], dtype=np.float32)
            embeddings.append(embedding)

        # Chuyển danh sách embeddings thành ma trận 2D
        embeddings_matrix = np.vstack(embeddings)
        
        # Lưu embeddings và FAISS index vào tệp
        np.save(embeddings_file, embeddings_matrix)
        faiss.write_index(index, index_file)
        print("Đã lưu embeddings và index vào tệp.")
    
    return embeddings_matrix, index

# Tạo và tải embeddings khi cần
embeddings_matrix, index = create_and_load_embeddings(data)

# Hàm truy xuất các nội dung liên quan từ FAISS
def retrieve_relevant_content(query: str, top_k: int = 5):
    try:
        # Chuyển câu hỏi thành embedding
        result = genai.embed_content(model="models/text-embedding-004", content=query)
        query_embedding = np.array(result['embedding'], dtype=np.float32)

        # Truy vấn FAISS để tìm các vector gần nhất
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

        # Trả về nội dung liên quan dựa trên các chỉ số (indices)
        relevant_content = []
        for idx in indices[0]:
            if idx < len(data):  # Ensure valid index
                relevant_content.append(create_combined_text(data[idx]))  # Rebuild the text from the section
            else:
                relevant_content.append("Nội dung không hợp lệ")

        return relevant_content

    except Exception as e:
        print(f"Đã xảy ra lỗi khi truy vấn: {e}")
        return []

# Hàm trả lời câu hỏi sử dụng mô hình tạo câu trả lời
def generate_answer(query: str, top_k: int = 5):
    try:
        # Truy xuất các đoạn văn liên quan
        relevant_content = retrieve_relevant_content(query, top_k)

        # Kết hợp các đoạn văn liên quan thành prompt cho mô hình sinh câu trả lời
        context = "\n".join(relevant_content)
        prompt = f"You are a helpful and informative bot that answers questions using text from the reference passage included below. \
                    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is not entirely related to the answer, you may ignore it.,Question: {query}\nRelated Data:\n{context}\nAnswer(Give answer precisely, do not hallucinate answer):"
        print(prompt)
        # Sử dụng mô hình tạo câu trả lời
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"Đã xảy ra lỗi khi tạo câu trả lời: {e}")
        return "Xin lỗi, tôi không thể trả lời câu hỏi của bạn."

# Giao diện Streamlit
st.title('About Presight')

# Kiểm tra nếu 'messages' chưa có trong session_state, nếu chưa thì khởi tạo
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn đã có trong session_state
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Lấy đầu vào của người dùng
prompt = st.chat_input('Message')

# Nếu có input, thêm tin nhắn vào session_state và hiển thị tin nhắn người dùng
if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Tạo câu trả lời
    answer = generate_answer(prompt, top_k=10)
    st.chat_message('assistant').markdown(answer)
    st.session_state.messages.append({'role': 'assistant', 'content': answer})
