# Crack_Finder
# Concrete Crack Detection with ResNet-18

This repository provides a pipeline for training a ResNet-18 neural network to classify concrete surface images as "cracked" or "uncracked". The model uses transfer learning, checkpointing, and a tqdm progress bar for efficient training on your own dataset.

---

## Features

- **Transfer Learning:** Utilizes a pretrained ResNet-18 model.
- **Custom Dataset Ready:** Designed to work with foldered image datasets (`Positive` and `Negative`).
- **Automatic Checkpointing:** Saves model and optimizer state at each epoch.
- **Progress Visualization:** Uses tqdm for real-time progress bars during training.

---

## Installation

First, create a Python environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then install the required packages:

```bash
pip install torch torchvision tqdm
```

---

## Dataset Structure

Your dataset should be organized as follows:

```
data_split/
    train/
        Positive/
            img001.jpg
            ...
        Negative/
            img051.jpg
            ...
    val/
        Positive/
            img101.jpg
            ...
        Negative/
            img151.jpg
            ...
```

- You can use the provided script or your own method to split your dataset into train/val sets.

---

## Training the Model

Save the training script (e.g., as `train.py`). Then, run:

```bash
python train.py
```

Model checkpoints will be saved in the `checkpoints` directory after each epoch. The final trained model weights will be saved as `crack_resnet18.pth`.

---

## Parameters

- **Batch Size:** 32 (can be changed in the script)
- **Image Size:** 227x227 pixels (to match reference datasets)
- **Epochs:** 3 (modify for longer training)
- **Learning Rate:** 0.001 (Adam optimizer)

---

## Using the Model

After training, you can load the model for inference or further fine-tuning:

```python
import torch
from torchvision import models

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('crack_resnet18.pth'))
model.eval()
```

---

## Notes

- For best results, ensure your validation set is representative.
- If using your own images, adjust the folder paths as needed in the script.

---

## License

This project is licensed under the MIT License.
