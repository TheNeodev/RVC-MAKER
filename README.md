<div align="center">
<img alt="LOGO" src="assets/ico.png" width="300" height="300" />

# Vietnamese RVC BY ANH
Công cụ chuyển đổi giọng nói chất lượng và hiệu suất cao đơn giản dành cho người Việt.

[![Vietnamese RVC](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)
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

- Chuyển đổi giọng nói (Chuyển đổi tệp/Chuyển đổi hàng loạt/Chuyển đổi với Whisper/Chuyển đổi văn bản)

- Chỉnh sửa nhạc nền

- Áp dụng hiệu ứng cho âm thanh

- Tạo dữ liệu huấn luyện (Từ đường dẫn liên kết)

- Huấn luyện mô hình (v1/v2, bộ mã hóa chất lượng cao)

- Dung hợp mô hình

- Đọc thông tin mô hình

- Xuất mô hình sang ONNX

- Tải xuống từ kho mô hình có sẳn

- Tìm kiếm mô hình từ web

- Trích xuất cao độ

- Hỗ trợ suy luận chuyển đổi âm thanh bằng mô hình ONNX

- Mô hình ONNX RVC cũng sẽ hỗ trợ chỉ mục để suy luận

- Nhiều tùy chọn mô hình:

F0: `pm, dio, mangio-crepe-tiny, mangio-crepe-small, mangio-crepe-medium, mangio-crepe-large, mangio-crepe-full, crepe-tiny, crepe-small, crepe-medium, crepe-large, crepe-full, fcpe, fcpe-legacy, rmvpe, rmvpe-legacy, harvest, yin, pyin, swipe`

F0_ONNX: Một số mô hình được chuyển đổi sang ONNX để hỗ trợ tăng tốc trích xuất

F0_HYBRID: Có thể kết hợp nhiều tùy chọn lại với nhau như `hybrid[rmvpe+harvest]` hoặc bạn có thể thử kết hợp toàn bộ tất cả tùy chọn lại với nhau

EMBEDDERS: `contentvec_base, hubert_base, japanese_hubert_base, korean_hubert_base, chinese_hubert_base, portuguese_hubert_base`

EMBEDDERS_ONNX: Tất cả mô hình nhúng ở trên điều có phiên bản được chuyển đổi sẳn sang ONNX để sử dụng tăng tốc trích xuất nhúng

EMBEDDERS_TRANSFORMERS: Tất cả mô hình nhúng ở trên điều có phiên bản được chuyển đổi sẳn sang huggingface để sử dụng thay thế cho fairseq

SPIN_EMBEDDERS: Một mô hình trích xuất nhúng mới, có thể mang đến chất lượng cao hơn các trích xuất cũ.

# Hướng dẫn sử dụng

**Sẽ có nếu tôi thực sự rảnh...**

# Cách cài đặt và sử dụng

- B1: **Cài đặt python từ trang chủ hoặc [python](https://www.python.org/ftp/python/3.10.7/python-3.10.7-amd64.exe) (YÊU CẦU PYTHON 3.10.x HOẶC PYTHON 3.11.x)**
- B2: **Cài đặt ffmpeg từ [FFMPEG](https://github.com/BtbN/FFmpeg-Builds/releases) giải nén và thêm vào PATH**
- B3: **Tải mã nguồn về và giải nén ra**
- B4: **Vào thư mục mã nguồn và mở Command Prompt hoặc Terminal**
- B5: **Nhập lệnh để cài đặt thư viện cần thiết để hoạt động**

```
python -m venv env
env\\Scripts\\activate
```

Nếu có GPU NVIDIA thì chạy bước này tùy theo cuda của bạn có thể thay đổi cu117 thành cu128...

```
# Nếu sử dụng Torch 2.3.1
python -m pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu117

# Nếu sử dụng Torch 2.6.0
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu117
```

Tiếp theo chạy

```
python -m pip install -r requirements.txt
```

- B5: **Chạy tệp run_app để mở giao diện sử dụng(Lưu ý: không tắt Command Prompt hoặc Terminal của giao diện)**
- Hoặc sử dụng cửa sổ Command Prompt hoặc cửa sổ Terminal trong thư mục mã nguồn
- Nếu muốn cho phép giao diện truy cập được các tệp ngoài dự án hãy thêm --allow_all_disk vào lệnh
```
env\\Scripts\\python.exe main\\app\\app.py --open
```

**Với trường hợp bạn sử dụng Tensorboard để kiểm tra huấn luyện**
```
Chạy tệp: tensorboard hoặc lệnh env\\Scripts\\python.exe main/app/tensorboard.py
```

# Sử dụng với cú pháp lệnh
```
python main\\app\\parser.py --help
```

# Cấu trúc chính của mã nguồn:

<pre>
Vietnamese-RVC-main
├── assets
│   ├── f0
│   ├── languages
│   │   ├── en-US.json
│   │   └── vi-VN.json
│   ├── logs
│   │   ├── mute
│   │   │   ├── f0
│   │   │   │   └── mute.wav.npy
│   │   │   ├── f0_voiced
│   │   │   │   └── mute.wav.npy
│   │   │   ├── sliced_audios
│   │   │   │   ├── mute32000.wav
│   │   │   │   ├── mute40000.wav
│   │   │   │   └── mute48000.wav
│   │   │   ├── sliced_audios_16k
│   │   │   │   └── mute.wav
│   │   │   ├── v1_extracted
│   │   │   │   └── mute.npy
│   │   │   └── v2_extracted
│   │   │       └── mute.npy
│   │   └── mute_spin
│   │       ├── f0
│   │       │   └── mute.wav.npy
│   │       ├── f0_voiced
│   │       │   └── mute.wav.npy
│   │       ├── sliced_audios
│   │       │   ├── mute32000.wav
│   │       │   ├── mute40000.wav
│   │       │   └── mute48000.wav
│   │       ├── sliced_audios_16k
│   │       │   └── mute.wav
│   │       ├── v1_extracted
│   │       │   └── mute.npy
│   │       └── v2_extracted
│   │           └── mute.npy
│   ├── models
│   │   ├── audioldm2
│   │   ├── embedders
│   │   ├── predictors
│   │   ├── pretrained_custom
│   │   ├── pretrained_v1
│   │   ├── pretrained_v2
│   │   ├── speaker_diarization
│   │   │   ├── assets
│   │   │   │   ├── gpt2.tiktoken
│   │   │   │   ├── mel_filters.npz
│   │   │   │   └── multilingual.tiktoken
│   │   │   └── models
│   │   └── uvr5
│   ├── presets
│   ├── weights
│   └── ico.png
├── audios
├── dataset
├── main
│   ├── app
│   │   ├── app.py
│   │   ├── tensorboard.py
│   │   └── parser.py
│   ├── configs
│   │   ├── v1
│   │   │   ├── 32000.json
│   │   │   ├── 40000.json
│   │   │   └── 48000.json
│   │   ├── v2
│   │   │   ├── 32000.json
│   │   │   ├── 40000.json
│   │   │   └── 48000.json
│   │   ├── config.json
│   │   └── config.py
│   ├── inference
│   │   ├── audio_effects.py
│   │   ├── audioldm2.py
│   │   ├── convert.py
│   │   ├── create_dataset.py
│   │   ├── create_index.py
│   │   ├── extract.py
│   │   ├── preprocess.py
│   │   ├── separator_music.py
│   │   └── train.py
│   ├── library
│   │   ├── algorithm
│   │   │   ├── commons.py
│   │   │   ├── modules.py
│   │   │   ├── mrf_hifigan.py
│   │   │   ├── onnx_export.py
│   │   │   ├── refinegan.py
│   │   │   ├── residuals.py
│   │   │   ├── separator.py
│   │   │   └── stftpitchshift.py
│   │   ├── architectures
│   │   │   ├── demucs_separator.py
│   │   │   ├── fairseq.py
│   │   │   └── mdx_separator.py
│   │   ├── audioldm2
│   │   │   ├── models.py
│   │   │   └── utils.py
│   │   ├── predictors
│   │   │   ├── CREPE.py
│   │   │   ├── FCPE.py
│   │   │   ├── RMVPE.py
│   │   │   ├── SWIPE.py
│   │   │   └── WORLD_WRAPPER.py
│   │   ├── speaker_diarization
│   │   │   ├── audio.py
│   │   │   ├── ECAPA_TDNN.py
│   │   │   ├── embedding.py
│   │   │   ├── encoder.py
│   │   │   ├── features.py
│   │   │   ├── parameter_transfer.py
│   │   │   ├── segment.py
│   │   │   ├── speechbrain.py
│   │   │   └── whisper.py
│   │   ├── uvr5_separator
│   │   │   ├── common_separator.py
│   │   │   ├── spec_utils.py
│   │   │   └── demucs
│   │   │       ├── apply.py
│   │   │       ├── demucs.py
│   │   │       ├── hdemucs.py
│   │   │       ├── htdemucs.py
│   │   │       ├── states.py
│   │   │       └── utils.py
│   │   └── utils.py
│   └── tools
│       ├── gdown.py
│       ├── huggingface.py
│       ├── mediafire.py
│       ├── meganz.py
│       ├── noisereduce.py
│       └── pixeldrain.py
├── docker-compose-cpu.yaml
├── docker-compose-cuda118.yaml
├── docker-compose-cuda128.yaml
├── Dockerfile
├── Dockerfile.cuda118
├── Dockerfile.cuda128
├── LICENSE
├── README.md
├── requirements.txt
├── run_app.bat
└── tensorboard.bat
</pre>

# LƯU Ý

- **Dự án này chỉ hỗ trợ trên gpu của NVIDIA (Có thể sẽ hỗ trợ AMD sau nếu tôi có gpu AMD để thử)**
- **Hiện tại các bộ mã hóa mới như MRF HIFIGAN vẫn chưa đầy đủ các bộ huấn luyện trước**
- **Bộ mã hóa MRF HIFIGAN và REFINEGAN không hỗ trợ huấn luyện khi không không huấn luyện cao độ**

# Điều khoản sử dụng

- Bạn phải đảm bảo rằng các nội dung âm thanh bạn tải lên và chuyển đổi qua dự án này không vi phạm quyền sở hữu trí tuệ của bên thứ ba.

- Không được phép sử dụng dự án này cho bất kỳ hoạt động nào bất hợp pháp, bao gồm nhưng không giới hạn ở việc sử dụng để lừa đảo, quấy rối, hay gây tổn hại đến người khác.

- Bạn chịu trách nhiệm hoàn toàn đối với bất kỳ thiệt hại nào phát sinh từ việc sử dụng sản phẩm không đúng cách.

- Tôi sẽ không chịu trách nhiệm với bất kỳ thiệt hại trực tiếp hoặc gián tiếp nào phát sinh từ việc sử dụng dự án này.

# Dự án này được xây dựng dựa trên các dự án như sau

- **[Applio](https://github.com/IAHispano/Applio/tree/main)** - IAHispano - MIT License
- **[Python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator/tree/main)** - Nomad Karaoke - MIT License
- **[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main)** - RVC Project - MIT License
- **[RVC-ONNX-INFER-BY-Anh](https://github.com/PhamHuynhAnh16/RVC_Onnx_Infer)** - Phạm Huỳnh Anh - MIT License
- **[Torch-Onnx-Crepe-By-Anh](https://github.com/PhamHuynhAnh16/TORCH-ONNX-CREPE)** - Phạm Huỳnh Anh - MIT License
- **[Hubert-No-Fairseq](https://github.com/PhamHuynhAnh16/hubert-no-fairseq)** - Phạm Huỳnh Anh - MIT License
- **[Local-attention](https://github.com/lucidrains/local-attention)** - Phil Wang - MIT License
- **[TorchFcpe](https://github.com/CNChTu/FCPE/tree/main)** - CN_ChiTu - MIT License
- **[FcpeONNX](https://github.com/deiteris/voice-changer/blob/master-custom/server/utils/fcpe_onnx.py)** - Yury - MIT License
- **[ContentVec](https://github.com/auspicious3000/contentvec)** - Kaizhi Qian - MIT License
- **[Mediafiredl](https://github.com/Gann4Life/mediafiredl)** - Santiago Ariel Mansilla - MIT License
- **[Noisereduce](https://github.com/timsainb/noisereduce)** - Tim Sainburg - MIT License
- **[World.py-By-Anh](https://github.com/PhamHuynhAnh16/world.py)** - Phạm Huỳnh Anh - MIT License
- **[Mega.py](https://github.com/odwyersoftware/mega.py)** - O'Dwyer Software - Apache 2.0 License
- **[Gdown](https://github.com/wkentaro/gdown)** - Kentaro Wada - MIT License
- **[Whisper](https://github.com/openai/whisper)** - OpenAI - MIT License
- **[PyannoteAudio](https://github.com/pyannote/pyannote-audio)** - pyannote - MIT License
- **[AudioEditingCode](https://github.com/HilaManor/AudioEditingCode)** - Hila Manor - MIT License
- **[StftPitchShift](https://github.com/jurihock/stftPitchShift)** - Jürgen Hock - MIT License
- **[Codename-RVC-Fork-3](https://github.com/codename0og/codename-rvc-fork-3)** - Codename;0 - MIT License

# Kho mô hình của công cụ tìm kiếm mô hình

- **[VOICE-MODELS.COM](https://voice-models.com/)**

# Các phương pháp trích xuất Pitch trong RVC

Tài liệu này trình bày chi tiết các phương pháp trích xuất cao độ được sử dụng, thông tin về ưu, nhược điểm, sức mạnh và độ tin cậy của từng phương pháp theo trải nghiệm cá nhân.

| Phương pháp        |      Loại      |          Ưu điểm          |            Hạn chế           |      Sức mạnh      |     Độ tin cậy     |
|--------------------|----------------|---------------------------|------------------------------|--------------------|--------------------|
| pm                 | Praat          | Nhanh                     | Kém chính xác                | Thấp               | Thấp               |
| dio                | PYWORLD        | Thích hợp với Rap         | Kém chính xác với tần số cao | Trung bình         | Trung bình         |
| harvest            | PYWORLD        | Chính xác hơn DIO         | Xử lý chậm hơn               | Cao                | Rất cao            |
| crepe              | Deep Learning  | Chính xác cao             | Yêu cầu GPU                  | Rất cao            | Rất cao            |
| mangio-crepe       | crepe finetune | Tối ưu hóa cho RVC        | Đôi khi kém crepe gốc        | Trung bình đến cao | Trung bình đến cao |
| fcpe               | Deep Learning  | Chính xác, thời gian thực | Cần GPU mạnh                 | Khá                | Trung bình         |
| fcpe-legacy        | Old            | Chính xác, thời gian thực | Cũ hơn                       | Khá                | Trung bình         |
| rmvpe              | Deep Learning  | Hiệu quả với giọng hát    | Tốn tài nguyên               | Rất cao            | Xuất sắc           |
| rmvpe-legacy       | Old            | Hỗ trợ hệ thống cũ        | Cũ hơn                       | Cao                | Khá                |
| yin                | Librosa        | Đơn giản, hiệu quả        | Dễ lỗi bội                   | Trung bình         | Thấp               |
| pyin               | Librosa        | Ổn định hơn YIN           | Tính toán phức tạp hơn       | Khá                | Khá                |
| swipe              | WORLD          | Độ chính xác cao          | Nhạy cảm với nhiễu           | Cao                | Khá                |

# Báo cáo lỗi

- **Với trường hợp gặp lỗi khi sử dụng mã nguồn này tôi thực sự xin lỗi bạn vì trải nghiệm không tốt này, bạn có thể gửi báo cáo lỗi thông qua cách phía dưới**
- **Bạn có thể báo cáo lỗi cho tôi thông qua hệ thống báo cáo lỗi webhook trong giao diện sử dụng**
- **Với trường hợp hệ thống báo cáo lỗi không hoạt động bạn có thể báo cáo lỗi cho tôi thông qua Discord `pham_huynh_anh` Hoặc [ISSUE](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/issues)**

# ☎️ Liên hệ tôi
- Discord: **pham_huynh_anh**