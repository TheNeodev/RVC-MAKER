import os
import tqdm
import requests

def HF_download_file(url, output_path=None):
    url = url.replace("/blob/", "/resolve/").strip()
    if output_path is None:
        output_path = os.path.basename(url)
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(url))

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        progress_bar = tqdm.tqdm(total=int(response.headers.get("content-length", 0)), desc=os.path.basename(url), unit="byte")
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
        return output_path
    else:
        raise ValueError(response.status_code)
