import os
import io
import sys
import onnx
import json
import torch
import onnxsim
import warnings

sys.path.append(os.getcwd())

from main.library.algorithm.synthesizers import SynthesizerONNX

warnings.filterwarnings("ignore")

def onnx_exporter(input_path, output_path, is_half=False, device="cpu"):
    cpt = (torch.load(input_path, map_location="cpu") if os.path.isfile(input_path) else None)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    model_name, model_author, epochs, steps, version, f0, model_hash, vocoder, creation_date = cpt.get("model_name", None), cpt.get("author", None), cpt.get("epoch", None), cpt.get("step", None), cpt.get("version", "v1"), cpt.get("f0", 1), cpt.get("model_hash", None), cpt.get("vocoder", "Default"), cpt.get("creation_date", None)
    text_enc_hidden_dim = 768 if version == "v2" else 256
    tgt_sr = cpt["config"][-1]

    net_g = SynthesizerONNX(*cpt["config"], use_f0=f0, text_enc_hidden_dim=text_enc_hidden_dim, vocoder=vocoder, checkpointing=False)
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(device)
    net_g = (net_g.half() if is_half else net_g.float())

    phone = torch.rand(1, 200, text_enc_hidden_dim).to(device)
    phone_length = torch.tensor([200]).long().to(device)
    ds = torch.LongTensor([0]).to(device)
    rnd = torch.rand(1, 192, 200).to(device)

    if f0:
        args = (phone, phone_length, ds, rnd, torch.randint(size=(1, 200), low=5, high=255).to(device), torch.rand(1, 200).to(device))
        input_names = ["phone", "phone_lengths", "ds", "rnd", "pitch", "pitchf"]
        dynamic_axes = {"phone": [1], "rnd": [2], "pitch": [1], "pitchf": [1]}
    else:
        args = (phone, phone_length, ds, rnd)
        input_names = ["phone", "phone_lengths", "ds", "rnd"]
        dynamic_axes = {"phone": [1], "rnd": [2]}

    with io.BytesIO() as model:
        torch.onnx.export(net_g, args, model, do_constant_folding=True, opset_version=17, verbose=False, input_names=input_names, output_names=["audio"], dynamic_axes=dynamic_axes)

        model, _ = onnxsim.simplify(onnx.load_model_from_string(model.getvalue()))
        model.metadata_props.append(onnx.StringStringEntryProto(key="model_info", value=json.dumps({"model_name": model_name, "author": model_author, "epoch": epochs, "step": steps, "version": version, "sr": tgt_sr, "f0": f0, "model_hash": model_hash, "creation_date": creation_date, "vocoder": vocoder, "text_enc_hidden_dim": text_enc_hidden_dim})))

    onnx.save(model, output_path)
    return output_path