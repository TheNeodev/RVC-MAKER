<div align="center">

# Vietnamese RVC BY ANH
Công cụ chuyển đổi giọng nói chất lượng và hiệu suất cao đơn giản dành cho người Việt.

[![Vietnamese RVC](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/drive/18Ed5HbwcX0di6aJymX0EaUNz-xXU5uUc?hl=vi#scrollTo=ers351v_CMGN)
[![Licence](https://img.shields.io/github/license/saltstack/salt?style=for-the-badge)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/blob/main/LICENSE)

</div>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AnhP/RVC-GUI)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/AnhP/Vietnamese-RVC-Project)

</div>

# Mô tả
Dự án này là một công cụ chuyển đổi giọng nói đơn giản, dễ sử dụng, được thiết kế cho người Việt Nam. Với mục tiêu tạo ra các sản phẩm chuyển đổi giọng nói chất lượng cao và hiệu suất tối ưu, dự án cho phép người dùng thay đổi giọng nói một cách mượt mà, tự nhiên.

# Các tính năng của dự án

- Tách nhạc (MDX-Net/Demucs)

- Chuyển đổi giọng nói (Chuyển đổi tệp/Chuyển đổi hàng loạt/Chuyển đổi văn bản)

- Áp dụng hiệu ứng cho âm thanh

- Tạo dữ liệu huấn luyện (Từ đường dẫn liên kết)

- Huấn luyện mô hình (v1/v2, bộ mã hóa chất lượng cao)

- Dung hợp mô hình

- Đọc thông tin mô hình

- Tải xuống từ kho mô hình có sẳn

- Tìm kiếm mô hình từ web

# Hướng dẫn sử dụng

**Sẽ có nếu tôi thực sự rảnh...**

# Cách cài đặt và sử dụng

- B1: **Cài đặt python từ trang chủ hoặc [python](https://www.python.org/ftp/python/3.10.7/python-3.10.7-amd64.exe) (YÊU CẦU PYTHON 3.10.x HOẶC PYTHON 3.11.x)**
- B2: **Cài đặt ffmpeg từ [FFMPEG](https://github.com/BtbN/FFmpeg-Builds/releases) giải nén và thêm vào PATH**
- B2: **Tải mã nguồn về và giải nén ra**
- B3: **Vào thư mục mã nguồn và mở Command Prompt hoặc Terminal**
- B4: **Nhập lệnh để cài đặt thư viện cần thiết để hoạt động**
```
python -m venv env
env\\Scripts\\activate
python -m pip install -r requirements.txt
```
- B5: **Chạy tệp run_app để mở giao diện sử dụng(Lưu ý: không tắt Command Prompt hoặc Terminal của giao diện)**
- Hoặc sử dụng cửa sổ Command Prompt hoặc cửa sổ Terminal trong thư mục mã nguồn
```
env\\Scripts\\python.exe main\\app\\app.py
```

**Với trường hợp bạn sử dụng Tensorboard để kiểm tra huấn luyện**
```
Chạy tệp: tensorboard hoặc lệnh env\\Scripts\\python.exe main/app/tensorboard.py
```

# Các đường dẫn thư mục chính của mã nguồn:

`assets\\languages`: **Thư mục chứa các tệp ngôn ngữ**

`assets\\logs`: **Thư mục chứa các tệp nhật ký và tệp chỉ mục mô hình**

`assets\\models\\embedders`: **Thư mục chứa các tệp mô hình nhúng**

`assets\\models\\predictors`: **Thư mục chứa một số tệp mô hình trích xuất dữ liệu của crepe, crepe-tiny, harvest, dio, rmvpe, fcpe**

`assets\\models\\pretrained_custom`: **Thư mục chứa các tệp mô hình huấn luyện trước tùy chỉnh**

`assets\\models\\pretrained_v1`: **Thư mục chứa các tệp mô hình huấn luyện trước v1**

`assets\\models\\pretrained_v2`: **Thư mục chứa các tệp mô hình huấn luyện trước v2**

`assets\\models\\uvr5`: **Thư mục chứa các tệp mô hình tách nhạc của Demucs và MDX**

`assets\\presets`: **Thư mục chứa các tệp cài đặt của chuyển đổi âm thanh**

`assets\\weights`: **Thư mục chứa các tệp mô hình**

`audios`: **Thư mục sẽ chứa các tệp âm thanh của bạn**

`dataset`: **Thư mục sẽ chứa các tệp dữ liệu âm thanh dùng cho việc huấn luyện mô hình**

# Các tệp tin cốt lỗi của mã nguồn

`main\\app\\app.py`: **Tệp tin hệ thống, giao diện của mã nguồn**

`main\\app\\tensorboard.py`: **Tệp tin hệ thống tensorboard**

`main\\configs\\v1`: **thư mục chứa các tệp cài đặt tốc độ lấy mẫu huấn luyện v1**

`main\\configs\\v2`: **thư mục chứa các tệp cài đặt tốc độ lấy mẫu huấn luyện v2**

`main\\configs\\config.json`: **Tệp tin cài đặt của giao diện**

`main\\configs\\config.py`: **Tệp khởi chạy các cài đặt**

`main\\inference\\audio_effects.py`: **Tệp tin thực hiện việc áp dụng hiệu ứng cho âm thanh**

`main\\inference\\convert.py`: **Tệp tin thực hiện xử lý và chuyển đổi âm thanh RVC**

`main\\inference\\create_dataset.py`: **Tệp tin thực hiện xử lý và tạo dữ liệu huấn luyện từ đường dẫn Youtube**

`main\\inference\\create_index.py`: **Tệp tin thực hiện việc tạo ra tệp tin chỉ mục**

`main\\inference\\extract.py`: **Tệp tin thực hiện việc trích xuất cao độ và trích xuất nhúng**

`main\\inference\\preprocess.py`: **Tệp tin thực hiện việc xử lý trước âm thanh dữ liệu huấn luyện trước khi trích xuất**

`main\\inference\\separator_music.py`: **Tệp tin thực hiện việc tách nhạc**

`main\\inference\\train.py`: **Tệp tin thực hiện việc huấn luyện mô hình RVC**

`main\\library\\algorithm\\commons.py`: **Tệp tin chức năng chung của RVC**

`main\\library\\algorithm\\modules.py`: **Tệp tin mô đun thuật toán sóng của RVC**

`main\\library\\algorithm\\mrf_hifigan.py`: **Tệp tin thuật toán của bộ mã hóa âm thanh MRF HIFIGAN**

`main\\library\\algorithm\\onnx_export.py`: **Tệp tin chuyển đổi mô hình RVC PYTORCH thành ONNX**

`main\\library\\algorithm\\refinegan.py`: **Tệp tin thuật toán của bộ mã hóa âm thanh REFINEGAN**

`main\\library\\algorithm\\residuals.py`: **Tệp tin chứa các lớp thuật toán như ResBlock,...**

`main\\library\\algorithm\\separator.py`: **Tệp tin thuật toán tách nhạc chính của DEMUCS\MDX**

`main\\library\\algorithm\\synthesizers.py`: **Tệp tin thuật toán tổng hợp**

`main\\library\\architectures\\demucs_separator.py`: **Tệp tin cấu trúc của bộ tách nhạc Demucs**

`main\\library\\architectures\\mdx_separator.py`: **Tệp tin cấu trúc của bộ tách nhạc MDX**

`main\\library\\predictors\\CREPE.py`: **Tệp tin bộ trích xuất cao độ F0 CREPE và CREPE-TINY**

`main\\library\\predictors\\FCPE.py`: **Tệp tin bộ trích xuất cao độ F0 FCPE**

`main\\library\\predictors\\RMVPE.py`: **Tệp tin bộ trích xuất cao độ F0 RMVPE**

`main\\library\\predictors\\WORLD.py`: **Tệp tin bộ trích xuất cao độ F0 HARVEST VÀ DIO**

`main\\library\\uvr5_separator\\demucs\\apply.py`: **Tệp tin áp dụng dành riêng cho DEMUCS**

`main\\library\\uvr5_separator\\demucs\\demucs.py`: **Tệp tin thư viện tách nhạc cho mô hình DEMUCS**

`main\\library\\uvr5_separator\\demucs\\hdemucs.py`: **Tệp tin thư viện tách nhạc cho mô hình HDEMUCS**

`main\\library\\uvr5_separator\\demucs\\htdemucs.py`: **Tệp tin thư viện tách nhạc cho mô hình HTDEMUCS**

`main\\library\\uvr5_separator\\demucs\\states.py`: **Tệp tin trạng thái dành riêng cho DEMUCS**

`main\\library\\uvr5_separator\\demucs\\utils.py`: **Tệp tin tiện ích dành riêng cho DEMUCS**

`main\\library\\uvr5_separator\\common_separator.py`: **Tệp tin chức năng chung của hệ thống tách nhạc MDX và DEMUCS**

`main\\library\\uvr5_separator\\spec_utils.py`: **Tệp tin thông số kỷ thuật của hệ thống tách nhạc**

`main\\library\\utils.py`: **Tệp tin chứa các tiện ích như: xử lý, tải âm thanh, kiểm tra và tải xuống mô hình thiếu**

`main\\tools\\edge_tts.py`: **Tệp tin công cụ chuyển đổi văn bản thành giọng nói của EDGE**

`main\\tools\\gdown.py`: **Tệp tin tải xuống tệp tin từ google drive**

`main\\tools\\google_tts.py`: **Tệp tin công cụ chuyển đổi văn bản thành giọng nói của google**

`main\\tools\\huggingface.py`: **Tệp tin tải xuống tệp tin từ huggingface**

`main\\tools\\mediafire.py`: **Tệp tin tải xuống tệp từ mediafire**

`main\\tools\\meganz.py`: **Tệp tin tải xuống tệp từ MegaNZ**

`main\\tools\\noisereduce.py`: **Tệp tin công cụ giảm tiếng ồn âm thanh**

`main\\tools\\pixeldrain.py`: **Tệp tin tải xuống tệp từ pixeldrain**

# LƯU Ý

- **Hiện tại các bộ mã hóa mới như MRF HIFIGAN và REFINEGAN vẫn chưa đầy đủ các bộ huấn luyện trước**
- **Bộ mã hóa MRF HIFIGAN và REFINEGAN không hỗ trợ huấn luyện khi không không huấn luyện cao độ**
- **MRF HIFIGAN và REFINEGAN chưa có bất cứ huấn luyện trước nào**

# Điều khoản sử dụng

- Bạn phải đảm bảo rằng các nội dung âm thanh bạn tải lên và chuyển đổi qua dự án này không vi phạm quyền sở hữu trí tuệ của bên thứ ba.

- Không được phép sử dụng dự án này cho bất kỳ hoạt động nào bất hợp pháp, bao gồm nhưng không giới hạn ở việc sử dụng để lừa đảo, quấy rối, hay gây tổn hại đến người khác.

- Bạn chịu trách nhiệm hoàn toàn đối với bất kỳ thiệt hại nào phát sinh từ việc sử dụng sản phẩm không đúng cách.

- Tôi sẽ không chịu trách nhiệm với bất kỳ thiệt hại trực tiếp hoặc gián tiếp nào phát sinh từ việc sử dụng dự án này.

# Dự án này dựa trên một số dự án chính như

- **[Applio](https://github.com/IAHispano/Applio/tree/main)**
- **[Python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator/tree/main)**
- **[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main)**

**và một số dự án khác**

- **[RVC-ONNX-INFER-BY-Anh](https://github.com/PhamHuynhAnh16/RVC_Onnx_Infer)**
- **[Torch-Onnx-Crepe-By-Anh](https://github.com/PhamHuynhAnh16/TORCH-ONNX-CREPE)**
- **[Local-attention](https://github.com/lucidrains/local-attention)**
- **[FcpeONNX](https://huggingface.co/deiteris/weights/tree/main)**
- **[ContentVec](https://github.com/auspicious3000/contentvec)**
- **[Mediafiredl](https://github.com/Gann4Life/mediafiredl)**
- **[Noisereduce](https://github.com/timsainb/noisereduce)**
- **[World.py-By-Anh](https://github.com/PhamHuynhAnh16/world.py)**
- **[Mega.py](https://github.com/odwyersoftware/mega.py)**
- **[Edge-TTS](https://github.com/rany2/edge-tts)**
- **[Gdown](https://github.com/wkentaro/gdown)**

# Kho mô hình của công cụ tìm kiếm mô hình

- **[VOICE-MODELS.COM](https://voice-models.com/)**

# Báo cáo lỗi

- **Với trường hợp gặp lỗi khi sử dụng mã nguồn này tôi thực sự xin lỗi bạn vì trải nghiệm không tốt này, bạn có thể gửi báo cáo lỗi thông qua cách phía dưới**
- **Bạn có thể báo cáo lỗi cho tôi thông qua hệ thống báo cáo lỗi webhook trong giao diện sử dụng**
- **Với trường hợp hệ thống báo cáo lỗi không hoạt động bạn có thể báo cáo lỗi cho tôi thông qua Discord `pham_huynh_anh` Hoặc [ISSUE](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/issues)**

# ☎️ Liên hệ tôi
- Discord: **pham_huynh_anh**