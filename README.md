🚀 How to Run the Application
Step 1: Install dependencies

Cài tất cả thư viện cần thiết từ requirements.txt:

pip install -r requirements.txt

Step 2: Run FastAPI (Backend Server)

Khởi động FastAPI server bằng uvicorn:

uvicorn app_api:app --reload

Server mặc định chạy tại:
👉 http://localhost:8000

API docs:
👉 http://localhost:8000/docs

Step 3: Run Streamlit (Frontend UI)

Mở terminal khác và chạy Streamlit:

streamlit run app/main.py
