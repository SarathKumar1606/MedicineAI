$ python main.py
2026-03-24 17:42:58.734757: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-03-24 17:43:02.909378: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

[INFO] Loaded Testing dataset
CSV: dataset\Testing\testing_labels.csv
Images: dataset\Testing\testing_words
Samples: 780
Augmentation: False
[INFO] Loading TrOCR model...
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
model.safetensors:  75%|████████████████████████████████████████████████████████▊                   | 996M/1.33G [05:32<02:28, 2.27MB/s]Error while downloading from https://huggingface.co/microsoft/trocr-base-handwritten/resolve/main/model.safetensors: HTTPSConnectionPool(host='cas-bridge.xethub.hf.co', port=443): Read timed out.
Trying to resume download...
model.safetensors:  75%|████████████████████████████████████████████████████████▊                   | 996M/1.33G [05:50<02:28, 2.27MB/s]Error while downloading from https://huggingface.co/microsoft/trocr-base-handwritten/resolve/main/model.safetensors: HTTPSConnectionPool(host='cas-bridge.xethub.hf.co', port=443): Read timed out.
Trying to resume download...
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████| 1.33G/1.33G [00:32<00:00, 3.26MB/s]
model.safetensors:  92%|█████████████████████████████████████████████████████████████████████      | 1.23G/1.33G [02:55<01:21, 1.31MB/s]
model.safetensors:  75%|████████████████████████████████████████████████████████▊                   | 996M/1.33G [08:44<03:05, 1.82MB/s]
C:\Users\rsara\AppData\Local\Programs\Python\Python313\Lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\rsara\.cache\huggingface\hub\models--microsoft--trocr-base-handwritten. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
Some weights of VisionEncoderDecoderModel were not initialized from the model checkpoint at microsoft/trocr-base-handwritten and are newly initialized: ['encoder.pooler.dense.bias', 'encoder.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 612kB/s]
[INFO] Model loaded on cpu

--- SAMPLE ---
Ground Truth: aceta
Image Path: dataset\Testing\testing_words\0.png

--- OCR RESULT ---
Prediction: acute .


“The pretrained TrOCR model was evaluated without fine-tuning and showed poor performance on domain-specific handwritten medical data.”

Model size: 1.33 GB
Slow download
CPU inference

👉 This confirms:

❌ Not suitable for deployment
