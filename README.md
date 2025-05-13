<div align="center">
<img alt="LOGO" src="assets/ico.png" width="300" height="300" />

# RVC MAKER
A high-quality voice conversion tool focused on ease of use and performance.

[![RVC MAKER](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TheNeodev/RVC-MAKER)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/TheNeodev/RVC-MAKER/blob/main/webui.ipynb)
[![Licence](https://img.shields.io/github/license/saltstack/salt?style=for-the-badge)](https://github.com/TheNeodev/RVC-MAKER/blob/main/LICENSE)

</div>

<div align="center">


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

# Usage Instructions

**Will be provided if I’m truly free...**

# Installation and Usage

- **Step 1**: Install Python from the official website or [Python](https://www.python.org/ftp/python/3.10.7/python-3.10.7-amd64.exe) (**REQUIRES PYTHON 3.10.x OR PYTHON 3.11.x**)
- **Step 2**: Install FFmpeg from [FFMPEG](https://github.com/BtbN/FFmpeg-Builds/releases), extract it, and add it to PATH
- **Step 3**: Download and extract the source code
- **Step 4**: Navigate to the source code directory and open Command Prompt or Terminal
- **Step 5**: Run the command to install the required libraries

python -m venv envenv\Scripts\activate

If you have an NVIDIA GPU, run this step depending on your CUDA version (you may need to change cu117 to cu128, etc.):

If using Torch 2.3.1
python -m pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu117
If using Torch 2.6.0
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu117

Then run:

python -m pip install -r requirements.txt

- **Step 6**: Run the `run_app` file to open the user interface (Note: Do not close the Command Prompt or Terminal for the interface)
- Alternatively, use Command Prompt or Terminal in the source code directory
- To allow the interface to access files outside the project, add `--allow_all_disk` to the command:

env\Scripts\python.exe main\app\app.py --open

**To use TensorBoard for training monitoring**:

Run the file: tensorboard or the command env\Scripts\python.exe main\app\tensorboard.py

# Command-Line Usage

python main\app\parser.py --help

</pre>

# NOTES

- **This project only supports NVIDIA GPUs (AMD support may be added later if I have an AMD GPU to test with)**
- **Currently, new encoders like MRF HIFIGAN do not yet have complete pre-trained datasets**
- **MRF HIFIGAN and REFINEGAN encoders do not support training without pitch training**

# Terms of Use

- You must ensure that the audio content you upload and convert through this project does not violate the intellectual property rights of third parties.

- The project must not be used for any illegal activities, including but not limited to fraud, harassment, or causing harm to others.

- You are solely responsible for any damages arising from improper use of the product.

- I will not be responsible for any direct or indirect damages arising from the use of this project.

# This Project is Built Based on the Following Projects


- **[Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)** -  PhamHuynhAnh16 - Apache License Version 2.0
                           
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

# Model Repository for Model Search Tool

- **[VOICE-MODELS.COM](https://voice-models.com/)**

# Pitch Extraction Methods in RVC

This document provides detailed information on the pitch extraction methods used, including their advantages, limitations, strengths, and reliability based on personal experience.

| Method            | Type           | Advantages                | Limitations                  | Strength           | Reliability        |
|-------------------|----------------|---------------------------|------------------------------|--------------------|--------------------|
| pm                | Praat          | Fast                      | Less accurate                | Low                | Low                |
| dio               | PYWORLD        | Suitable for rap          | Less accurate at high frequencies | Medium             | Medium             |
| harvest           | PYWORLD        | More accurate than DIO    | Slower processing            | High               | Very high          |
| crepe             | Deep Learning  | High accuracy             | Requires GPU                 | Very high          | Very high          |
| mangio-crepe      | Crepe finetune | Optimized for RVC         | Sometimes less accurate than original crepe | Medium to high     | Medium to high     |
| fcpe              | Deep Learning  | Accurate, real-time       | Requires powerful GPU        | Good               | Medium             |
| fcpe-legacy       | Old            | Accurate, real-time       | Older                        | Good               | Medium             |
| rmvpe             | Deep Learning  | Effective for singing voices | Resource-intensive           | Very high          | Excellent          |
| rmvpe-legacy      | Old            | Supports older systems     | Older                        | High               | Good               |
| yin               | Librosa        | Simple, efficient         | Prone to octave errors       | Medium             | Low                |
| pyin              | Librosa        | More stable than YIN      | More complex computation     | Good               | Good               |
| swipe             | WORLD          | High accuracy             | Sensitive to noise           | High               | Good               |

# Bug Reporting

- **If you encounter an error while using this source code, I sincerely apologize for the poor experience. You can report the bug using the methods below.**
- **You can report bugs to me via the webhook bug reporting system in the user interface.**
- **If the bug reporting system is not working, you can report bugs to me via Discord `pham_huynh_anh` or [ISSUE](https://github.com/theneodev/RVC-MAKER/issues).**

