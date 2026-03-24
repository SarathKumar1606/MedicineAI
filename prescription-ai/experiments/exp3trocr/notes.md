hug

💊 Matching medicines...

----- MATCHED MEDICINES -----

- ace

✅ Results saved to: outputs/predictions\prescription.txt

rsara@Vivobook_16 MINGW64 /d/Deep Learning Project/prescription-ai (main)
$ python main.py --image prescription.png
2026-03-24 22:37:55.929062: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-03-24 22:38:00.285010: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Using CPU. Note: This module is much faster with a GPU.
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Some weights of VisionEncoderDecoderModel were not initialized from the model checkpoint at microsoft/trocr-base-handwritten and are newly initialized: ['encoder.pooler.dense.bias', 'encoder.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

📄 Processing: prescription.png

🔍 Running OCR...
C:\Users\rsara\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)

----- OCR TEXT -----

Sutter Health
Prescription for Health & Happiness
A dose ofjoy and self-compassion
r. Leaf class .
iso from
date :
Provider:
SuzeS
Prescribed to:
R spreud the
love .
deep'
Take a few
breaths
Call an old friend
hug
Give someone a
watched a sunset .
Help a friend in need
Write a thank you note
singi
in the shower
Dance to your favorite song
mass . Dr. 2 ) 3/ Progress
Go for a walk in a beautiful place
Forgive someone
XAs Needed
Talk to yourself with a kinder voice
03 04
Refills:

💊 Matching medicines...

----- MATCHED MEDICINES -----

- ace

✅ Results saved to: outputs/predictions\prescription.txt

rsara@Vivobook_16 MINGW64 /d/Deep Learning Project/prescription-ai (main)