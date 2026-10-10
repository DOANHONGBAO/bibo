# BIBO — Personal Research Memory Agent

BIBO là dự án xây dựng trợ lý ghi nhớ và truy xuất tri thức nghiên cứu cá nhân. **Giai đoạn đang triển khai:** Document Retrieval MVP — nhập tài liệu cục bộ, chia đoạn, tạo embedding và tìm lại nguyên văn đoạn liên quan. Bản này chưa có agent, chatbot, sinh câu trả lời hay ký ức dài hạn. [PROJECT_REQUIREMENTS.txt](PROJECT_REQUIREMENTS.txt) mô tả tầm nhìn rộng hơn; MVP này cố ý chỉ triển khai phần truy xuất PDF/TXT/Markdown.

## Kiến trúc hiện tại

```text
Tệp PDF/TXT/Markdown → Parser → Chunker → BGE-M3 → ChromaDB (data/chroma)
Truy vấn Việt/Anh ─────────────────→ BGE-M3 → Tìm top-k → Đoạn gốc + nguồn
                         CLI: prepare-model, ingest, ingest-dir, search,
                              interactive, benchmark, list, stats, remove
```

Parser giữ số trang PDF và đường dẫn nguồn; chunker không trộn các trang PDF. `vector_store.py` là lớp duy nhất gọi ChromaDB. Tệp gốc không bị sửa hoặc xóa. Cùng một mô hình BGE-M3 mã hóa tài liệu và truy vấn thành vector 1024 chiều chuẩn hóa; Chroma dùng khoảng cách cosine, còn CLI hiển thị `1 - distance` như cosine similarity.

## Môi trường phát triển và cài đặt

Yêu cầu Conda và Python 3.11. Trên Windows, kích hoạt môi trường bằng:

```powershell
conda activate bibo
python --version
python -m pip install -e ".[test]"
```

`python --version` **phải** là 3.11. Nếu vẫn hiện Python 3.14 từ `uv`, PowerShell chưa kích hoạt Conda thật sự dù `conda activate` không báo lỗi. Trong phiên PowerShell hiện tại, chạy:

```powershell
. "C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1"
conda activate bibo
Get-Command python
python --version
```

Nếu không muốn thay đổi phiên shell, gọi trực tiếp Python đã được kiểm tra trong môi trường `bibo`:

```powershell
$BiboPython = Join-Path $env:USERPROFILE ".conda\envs\bibo\python.exe"
& $BiboPython --version
& $BiboPython -m bibo.cli ingest-dir "documents"
```

Đường dẫn môi trường trên là vị trí Conda hiện dùng trên máy này; nếu bạn tạo môi trường ở nơi khác, thay bằng đường dẫn `python.exe` của môi trường đó. Có thể chạy `conda init powershell` rồi **mở lại PowerShell** để kích hoạt tự động trong các phiên sau, hoặc dùng Anaconda Prompt. Nếu môi trường chưa có, chạy `conda create -n bibo python=3.11 -y` trước. Không cần khóa API hay gửi tài liệu tới dịch vụ ngoài.

### Chuẩn bị mô hình một lần

```powershell
python -m bibo.cli prepare-model
python -m bibo.cli prepare-model --offline
```

Lệnh đầu tải BGE-M3 công khai khi cache chưa đủ; lần đầu cần mạng và khoảng vài GB dung lượng. Nếu mô hình đã có, lệnh chỉ xác minh cache. Lệnh `--offline` tuyệt đối không tải. Các lệnh nhập và tìm kiếm chỉ nạp từ đường dẫn local đã xác minh, nên không phát sinh yêu cầu metadata Hugging Face mỗi lần chạy. Mặc định mô hình ở `data/model_cache/`; có thể đặt `BIBO_MODEL_CACHE` hoặc trỏ `BIBO_MODEL_DIR` tới thư mục mô hình hoàn chỉnh. Nếu cache chưa đủ, BIBO báo cần chạy `prepare-model`; không tự tải trong lúc truy vấn.

## Sử dụng

Định dạng hỗ trợ: `.pdf` có lớp văn bản, `.txt` UTF-8 và `.md` UTF-8. Có thể đặt tài liệu riêng trong `documents/` (được Git bỏ qua). Từ thư mục dự án:

```powershell
python -m bibo.cli ingest "documents\paper.pdf"
python -m bibo.cli ingest-dir "documents"
python -m bibo.cli search "Các phương pháp giảm false positive" --top-k 5
python -m bibo.cli search "How can false positives be reduced?"
python -m bibo.cli interactive --top-k 5
python -m bibo.cli list
python -m bibo.cli stats
python -m bibo.cli remove DOCUMENT_ID
```

Trong `interactive`, gõ nhiều câu hỏi rồi `exit` hoặc `quit` để thoát. Mô hình và ChromaDB chỉ được mở một lần cho phiên đó; lệnh `search` chạy riêng vẫn phải nạp mô hình trong tiến trình mới.

`ingest-dir` tìm đệ quy và bỏ qua loại tệp không hỗ trợ. Cùng nội dung tệp (SHA-256) đã hoàn tất sẽ được **bỏ qua trước khi đọc PDF hoặc tạo embedding**. Nếu lần trước ghi dở, BIBO chỉ bổ sung chunk còn thiếu và giữ các vector đã có. PDF đã phân tích được lưu cache cục bộ theo hash tệp, phiên bản parser và cấu hình. Khi nội dung tệp đổi, hash/ID mới được lập chỉ mục riêng; bản cũ còn lại để người dùng xem và xóa thủ công. `remove` yêu cầu xác nhận; dùng `--yes` trong script. Lệnh này chỉ xóa chỉ mục, không xóa tệp nguồn.

Cache văn bản PDF trích xuất, vector và trọng số mô hình đều ở thư mục cục bộ được Git bỏ qua. Không đưa `data/`, `models/` hoặc `documents/` lên GitHub.

Ví dụ đầu ra tìm kiếm (điểm chỉ minh họa):

```text
Rank 1 | Document: paper.pdf | Page: 5 | Cosine similarity: 0.8700
Source: D:\research\paper.pdf | Document ID: ...
Content:
Original retrieved passage...
```

## Cấu hình

Biến môi trường tùy chọn:

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `BIBO_DATA_DIR` | `data/` trong dự án | Nơi lưu ChromaDB và cache mặc định |
| `BIBO_MODEL_CACHE` | `data/model_cache/` | Cache mô hình tải về |
| `BIBO_MODEL_DIR` | Không đặt | Thư mục mô hình hoàn chỉnh dùng trực tiếp, không truy cập mạng |
| `BIBO_CHUNK_SIZE_WORDS` | `500` | Số **từ** tối đa gần đúng mỗi đoạn |
| `BIBO_CHUNK_OVERLAP_WORDS` | `75` | Số **từ** lặp giữa hai đoạn |
| `BIBO_BATCH_SIZE` | `4` | Số đoạn cho mỗi batch embedding và ghi chỉ mục |

Chunker dùng chuỗi không phải khoảng trắng làm đơn vị **từ**, không dùng tokenizer nên các giá trị 500/75 **không phải token**. Nó ưu tiên ngắt ở cuối đoạn/câu gần giới hạn, giữ thứ tự và vùng lặp; mỗi trang PDF được xử lý riêng. Mô hình, số chiều và cấu hình chia đoạn được ghi trong collection; nếu thay cấu hình trên chỉ mục cũ, chương trình báo lỗi. Dùng một `BIBO_DATA_DIR` khác hoặc cấu hình ban đầu để tránh trộn chỉ mục.

Batch 4 là mặc định đo được nhanh nhất trên máy CPU hiện tại cho PDF kiểm thử 16 đoạn; `BIBO_BATCH_SIZE` có thể điều chỉnh theo máy. Batch size không làm thay đổi lược đồ chỉ mục. BIBO đếm token thực trước khi mã hóa và báo lỗi nếu một chunk vượt giới hạn mô hình; nó không tự cắt bỏ nội dung nguồn.

## Đo hiệu năng

```powershell
python -m bibo.cli ingest "documents\paper.pdf" --profile
python -m bibo.cli search "research question" --profile
python -m bibo.cli interactive --profile
python -m bibo.cli benchmark
python -m bibo.cli benchmark --document "documents\paper.pdf"
```

`benchmark` mặc định tạo PDF tổng hợp cố định; có thể chọn PDF cục bộ để đo tải thực tế. Lệnh đo hash, đọc PDF, cache, chia đoạn, token hóa, tải mô hình, các batch 4/8/16, ghi Chroma, truy vấn đầu/ấm và một lệnh tìm kiếm mới. Báo cáo JSON nằm ở `benchmarks/results/` và không được commit; không chứa văn bản tài liệu. Lệnh dùng chỉ mục riêng trong `data/benchmark_runs/`, có lưu bản sao đoạn văn cục bộ để tái kiểm tra và không sửa chỉ mục nghiên cứu. GPU/FP16 chỉ được đánh giá khi PyTorch phát hiện CUDA phù hợp.

Phương pháp và kết quả trên máy hiện tại: [benchmarks/README.md](benchmarks/README.md).

## Kiểm thử

```powershell
python -m pytest -q
```

Kiểm thử tự động dùng mô hình giả cố định nên không tải BGE-M3. Chúng kiểm tra parser, chia đoạn, lưu trữ bền vững, metadata, phát hiện trùng, thứ hạng và truy vấn trên chỉ mục rỗng. Tài liệu mẫu tổng hợp nằm ở `tests/fixtures/sample.md`.

## Giới hạn và hướng mở rộng

- PDF quét không có lớp văn bản cần OCR; phiên bản này sẽ báo lỗi rõ ràng.
- Chưa hỗ trợ DOCX, lọc theo dự án, RAG/LLM, agent, MCP, API hay giao diện web.
- Tìm kiếm đa ngôn ngữ dựa trên BGE-M3; chất lượng Việt–Anh cần được đánh giá trên bộ dữ liệu nghiên cứu thực tế, không thể suy ra từ ví dụ tổng hợp.
- Đổi nội dung tệp tạo ID mới theo hash; bản cũ phải được xóa khỏi chỉ mục nếu không còn cần.
- Hướng mở rộng: OCR, DOCX, đánh giá Recall@k/MRR/nDCG, quản lý dự án và phê duyệt ký ức theo các giai đoạn trong tài liệu yêu cầu.
