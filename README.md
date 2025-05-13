<div align="center">
<img alt="LOGO" src="assets/ico.png" width="300" height="300" />

# RVC MAKER
A high-quality voice conversion tool focused on ease of use and performance.

[![RVC MAKER](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)
[![Licence](https://img.shields.io/github/license/saltstack/salt?style=for-the-badge)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/blob/main/LICENSE)

</div>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AnhP/RVC-GUI)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/AnhP/Vietnamese-RVC-Project)

</div>

# Description
This project is a simple, easy-to-use voice conversion tool designed for Vietnamese users. With the goal of creating high-quality and high-performance voice conversion products, the project allows users to change voices smoothly and naturally.

# Project Features

- Music separation (MDX-Net/Demucs)

- Voice conversion (File conversion/Batch conversion/Conversion with Whisper/Text-to-speech conversion)

- Background music editing

- Apply effects to audio

- Generate training data (From linked paths)

- Model training (v1/v2, high-quality encoders)

- Model fusion

- Read model information

- Export models to ONNX

- Download from pre-existing model repositories

- Search for models on the web

- Pitch extraction

- Support for audio conversion inference using ONNX models

- ONNX RVC models also support indexing for inference

- Multiple model options:

**F0**: `pm, dio, mangio-crepe-tiny, mangio-crepe-small, mangio-crepe-medium, mangio-crepe-large, mangio-crepe-full, crepe-tiny, crepe-small, crepe-medium, crepe-large, crepe-full, fcpe, fcpe-legacy, rmvpe, rmvpe-legacy, harvest, yin, pyin, swipe`

**F0_ONNX**: Some models are converted to ONNX to support accelerated extraction

**F0_HYBRID**: Multiple options can be combined, such as `hybrid[rmvpe+harvest]`, or you can try combining all options together

**EMBEDDERS**: `contentvec_base, hubert_base, japanese_hubert_base, korean_hubert_base, chinese_hubert_base, portuguese_hubert_base`

**EMBEDDERS_ONNX**: All the above embedding models have ONNX versions pre-converted for accelerated embedding extraction

**EMBEDDERS_TRANSFORMERS**: All the above embedding models have versions pre-converted to Hugging Face for use as an alternative to Fairseq

**SPIN_EMBEDDERS**: A new embedding extraction model that may provide higher quality than older extractions

# Installation and Usage

- **Step 1**: Install Python from the official website or [Python](https://www.python.org/ftp/python/3.10.7/python-3.10.7-amd64.exe) (**REQUIRES PYTHON 3.10.x OR PYTHON 3.11.x**)
- **Step 2**: Install FFmpeg from [FFMPEG](https://github.com/BtbN/FFmpeg-Builds/releases), extract it, and add it to PATH
- **Step 3**: Download and extract the source code
- **Step 4**: Navigate to the source code directory and open Command Prompt or Terminal
- **Step 5**: Run the command to install the required libraries
