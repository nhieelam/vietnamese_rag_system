# Rerank — `RerankerService` (Cross-Encoder)

Tài liệu mô tả **re-ranking** sau bước truy hồi ban đầu, **mô hình dùng**, **cách bật/tắt**, và **vị trí trong pipeline** RAG / Co-RAG / Self-RAG.

## Vì sao cần rerank?

**Truy hồi lần đầu** (FAISS similarity, hoặc hybrid BM25 + vector) dùng **mô hình bi-encoder** / khoảng cách: nhanh nhưng **câu hỏi** và **đoạn văn** được encode **độc lập** — thứ hạng đôi khi chưa khớp “đúng cặp câu hỏi–đoạn văn” theo nghĩa ngôn ngữ sâu.

**Cross-encoder** nhận cặp `(query, đoạn)` trong **một** lượt tính toán, nên chất lượng xếp hạng tốt hơn nhưng **chậm hơn** và chỉ nên chạy trên **một tập ứng viên hữu hạn** (top sau retrieve).

## Thư viện và mô hình

| Thành phần | Chi tiết |
|------------|----------|
| Lớp | [`RerankerService`](../app/services/reranker_service.py) |
| Thư viện | **sentence-transformers** — `CrossEncoder` |
| Model mặc định | `BAAI/bge-reranker-v2-m3` (đa ngôn ngữ, hằng `MODEL_NAME` trong file service) |
| `max_length` | `512` token khi load `CrossEncoder` |

> **Lưu ý:** Đây là **mô hình riêng** với mô hình **embedding** dùng cho FAISS (`AppConfig.EMBEDDING_MODEL_NAME`). Rerank **không** thay thế bước tạo vector index — chỉ sắp xếp lại danh sách ứng viên đã lấy. Xem [EMBEDDING.md](./EMBEDDING.md).

## API

### `get_model()`

- Tải **lười** (lần đầu cần mới load).
- Ưu tiên **`@st.cache_resource`** (Streamlit) để tránh tải lại mỗi rerun; nếu không có context Streamlit, load `CrossEncoder` trực tiếp.
- Cache module-level (`_MODEL_CACHE`) giữ tham chiếu model.

### `rerank(query, docs_with_scores, top_k=None)`

- **Đầu vào:** `List[Tuple[Document, float]]` — danh sách ứng viên kèm điểm từ bước trước (FAISS/hybrid; điểm cũ không dùng lại cho xếp hạng cuối, chỉ để giữ cặp doc).
- Gọi `CrossEncoder.predict` trên các cặp `(query, doc.page_content)`.
- **Chuẩn hóa** điểm thô bằng **sigmoid** về khoảng **0..1** để thống nhất với pipeline **citation** (hiển thị relevance).
- Sắp xếp **giảm dần** theo điểm mới; cắt `top_k` nếu có.
- Nếu **lỗi** load model hoặc `predict`, trả về **thứ tự gốc** (cắt theo `top_k` nếu có) để pipeline không bị dừng hẳn.

## Bật / tắt trong ứng dụng

- Sidebar → **Chunk & Retrieval** → checkbox **“Enable Cross-Encoder Re-ranking”** (`session` key: `use_reranker`).
- `SessionService.get_use_reranker()` đọc cờ này.

Khi bật rerank, các service thường **tăng số ứng viên** trước khi cắt (ví dụ `fetch_k` lớn hơn) rồi mới `rerank(..., top_k=k)` — tham số tương ứng trong `AppConfig`: `RERANK_CANDIDATE_MULTIPLIER`, `RERANK_CANDIDATE_MIN` (RAG; Self-RAG có công thức tương tự trong code).

## Dùng ở đâu (sau khi đã `retrieve` / score)

| Module | Hành vi |
|--------|---------|
| [`rag_service.py`](../app/services/rag_service.py) | Nếu `use_reranker` và có `scored` → `RerankerService.rerank` với câu hỏi (đã rephrase theo lịch sử nếu có) và `top_k = k` retrieval. |
| [`co_rag_service.py`](../app/services/co_rag_service.py) | Sau gom kết quả nhiều sub-query, rerank theo câu standalone. |
| [`self_rag_service.py`](../app/services/self_rag_service.py) | Trong `_retrieve`, sau bước retrieve + diversify, rerank tùy cờ. |

**Không** gọi `RerankerService` khi chưa có danh sách ứng viên hoặc khi tắt checkbox.

## So sánh nhanh

| Bước | Vai trò |
|------|--------|
| Embedding + FAISS | Index & tìm nhanh theo tương tự vector — [VECTOR.md](./VECTOR.md) |
| Hybrid (BM25 + vector) | Kết hợp từ khóa + ngữ nghĩa — [HYBRID_RETRIEVER.md](./HYBRID_RETRIEVER.md) |
| **Rerank** | Tinh chỉnh thứ tự ứng viên bằng cross-encoder trên tập nhỏ |

## Tài liệu liên quan

- Trích dẫn sau cùng dùng `Citation` + điểm — [CITATION.md](./CITATION.md)
- Tổng thể retrieval — [ARCHITECTURE.md](./ARCHITECTURE.md)
