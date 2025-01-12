import os
import json
import torch

version_config_paths = [os.path.join(version, size) for version in ["v1", "v2"] for size in ["32000.json", "40000.json", "44100.json", "48000.json"]]

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances: instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_name = (torch.cuda.get_device_name(int(self.device.split(":")[-1])) if self.device.startswith("cuda") else None)
        self.translations = self.multi_language()
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()
    
    def multi_language(self):
        try:
            with open(os.path.join("main", "configs", "config.json"), "r") as f:
                configs = json.load(f)

            lang = configs.get("language", "vi-VN")
            if len([l for l in os.listdir(os.path.join("assets", "languages")) if l.endswith(".json")]) < 1: raise FileNotFoundError("Không tìm thấy bất cứ gói ngôn ngữ nào(No package languages found)")

            if not lang: lang = "vi-VN"
            if lang not in configs["support_language"]: raise ValueError("Ngôn ngữ không được hỗ trợ(Language not supported)")

            lang_path = os.path.join("assets", "languages", f"{lang}.json")
            if not os.path.exists(lang_path): lang_path = os.path.join("assets", "languages", "vi-VN.json")

            with open(lang_path, encoding="utf-8") as f:
                translations = json.load(f)
        except json.JSONDecodeError:
            print(self.translations["empty_json"].format(file=lang))
            pass
        return translations

    def load_config_json(self):
        configs = {}
        for config_file in version_config_paths:
            try:
                with open(os.path.join("main", "configs", config_file), "r") as f:
                    configs[config_file] = json.load(f)
            except json.JSONDecodeError:
                print(self.translations["empty_json"].format(file=config_file))
                pass
        return configs

    def device_config(self):
        if self.device.startswith("cuda"): self.set_cuda_config()
        elif self.has_mps(): self.device = "mps"
        else: self.device = "cpu"

        if self.gpu_mem is not None and self.gpu_mem <= 4: return 1, 5, 30, 32
        return 1, 6, 38, 41

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024**3)

    def has_mps(self):
        return torch.backends.mps.is_available()