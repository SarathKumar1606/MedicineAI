[INFO] Loaded Training dataset
CSV: dataset/Training/training_labels.csv
Images: dataset/Training/training_words
Samples: 3120
Augmentation: True
Epoch 1/15: 100%|██████████| 98/98 [00:14<00:00,  6.73it/s, loss=3.23]
Epoch 1 Loss: 450.5310
Epoch 2/15: 100%|██████████| 98/98 [00:13<00:00,  7.43it/s, loss=3.07]
Epoch 2 Loss: 311.3163
Epoch 3/15: 100%|██████████| 98/98 [00:14<00:00,  6.81it/s, loss=3.14]
Epoch 3 Loss: 306.5170
Epoch 4/15: 100%|██████████| 98/98 [00:13<00:00,  7.35it/s, loss=3.19]
Epoch 4 Loss: 303.9755
Epoch 5/15: 100%|██████████| 98/98 [00:13<00:00,  7.31it/s, loss=3.07]
Epoch 5 Loss: 301.4716
Epoch 6/15: 100%|██████████| 98/98 [00:13<00:00,  7.28it/s, loss=3.09]
Epoch 6 Loss: 299.4534
Epoch 7/15: 100%|██████████| 98/98 [00:14<00:00,  6.98it/s, loss=3.1]
Epoch 7 Loss: 297.5875
Epoch 8/15: 100%|██████████| 98/98 [00:13<00:00,  7.07it/s, loss=2.89]
Epoch 8 Loss: 294.8758
Epoch 9/15: 100%|██████████| 98/98 [00:13<00:00,  7.03it/s, loss=3.12]
Epoch 9 Loss: 291.9670
Epoch 10/15: 100%|██████████| 98/98 [00:14<00:00,  6.95it/s, loss=2.87]
Epoch 10 Loss: 287.9765
Epoch 11/15: 100%|██████████| 98/98 [00:14<00:00,  6.84it/s, loss=2.87]
Epoch 11 Loss: 283.9935
Epoch 12/15: 100%|██████████| 98/98 [00:14<00:00,  6.88it/s, loss=3.14]
Epoch 12 Loss: 280.3655
Epoch 13/15: 100%|██████████| 98/98 [00:13<00:00,  7.02it/s, loss=2.68]
Epoch 13 Loss: 277.1714
Epoch 14/15: 100%|██████████| 98/98 [00:13<00:00,  7.06it/s, loss=2.86]
Epoch 14 Loss: 275.7188
Epoch 15/15: 100%|██████████| 98/98 [00:14<00:00,  6.99it/s, loss=2.78]Epoch 15 Loss: 274.4068
Model saved!



rsara@Vivobook_16 MINGW64 /d/Deep Learning Project/prescription-ai (main)
$ python main.py
[INFO] CRNN model loaded.

Prediction: ani





[INFO] Loaded Training dataset
CSV: dataset/Training/training_labels.csv
Images: dataset/Training/training_words
Samples: 3120
Augmentation: True
FineTune Epoch 1/15: 100%|██████████| 98/98 [00:13<00:00,  7.16it/s, loss=2.74]
Epoch 1 Loss: 265.2194
FineTune Epoch 2/15: 100%|██████████| 98/98 [00:13<00:00,  7.11it/s, loss=2.64]
Epoch 2 Loss: 264.9955
FineTune Epoch 3/15: 100%|██████████| 98/98 [00:14<00:00,  6.99it/s, loss=2.74]
Epoch 3 Loss: 265.0254
FineTune Epoch 4/15: 100%|██████████| 98/98 [00:14<00:00,  6.89it/s, loss=2.83]
Epoch 4 Loss: 264.9738
FineTune Epoch 5/15: 100%|██████████| 98/98 [00:14<00:00,  6.87it/s, loss=2.56]
Epoch 5 Loss: 264.8335
FineTune Epoch 6/15: 100%|██████████| 98/98 [00:14<00:00,  6.87it/s, loss=2.57]
Epoch 6 Loss: 264.8451
FineTune Epoch 7/15: 100%|██████████| 98/98 [00:13<00:00,  7.02it/s, loss=2.64]
Epoch 7 Loss: 264.8274
FineTune Epoch 8/15: 100%|██████████| 98/98 [00:13<00:00,  7.08it/s, loss=2.62]
Epoch 8 Loss: 264.8112
FineTune Epoch 9/15: 100%|██████████| 98/98 [00:13<00:00,  7.00it/s, loss=2.72]
Epoch 9 Loss: 264.0800
FineTune Epoch 10/15: 100%|██████████| 98/98 [00:14<00:00,  6.91it/s, loss=2.57]
Epoch 10 Loss: 263.2988
FineTune Epoch 11/15: 100%|██████████| 98/98 [00:14<00:00,  6.85it/s, loss=2.93]
Epoch 11 Loss: 262.5385
FineTune Epoch 12/15: 100%|██████████| 98/98 [00:14<00:00,  6.96it/s, loss=2.57]
Epoch 12 Loss: 261.8975
FineTune Epoch 13/15: 100%|██████████| 98/98 [00:14<00:00,  6.97it/s, loss=2.68]
Epoch 13 Loss: 261.2962
FineTune Epoch 14/15: 100%|██████████| 98/98 [00:14<00:00,  6.99it/s, loss=2.58]
Epoch 14 Loss: 260.7379
FineTune Epoch 15/15: 100%|██████████| 98/98 [00:13<00:00,  7.03it/s, loss=2.63]Epoch 15 Loss: 259.9018

✅ Fine-tuning complete. Model saved as crnn_finetuned.pth

$ python main.py
[INFO] CRNN model loaded.

Prediction: ana