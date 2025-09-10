# Concrete Crack Detection with ResNet-18

## Introduction

This repository provides a complete pipeline for detecting and localizing cracks in concrete using a fine-tuned ResNet-18 neural network. Simply provide a photo or even a video of the inspected surface—the model will detect cracks, highlight their locations, and output their geographical coordinates in GeoJSON format.

The model is trained for 3 epochs on a dataset of cracks from the [Kaggle Surface Crack Detection Dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection), where each sample is a 227x227 pixel image labeled as cracked or uncracked concrete.

## How It Works

The detection script divides the input image into a grid, where each cell matches the size of the training images (227x227 pixels). Each cell is independently classified as "crack" or "no crack." Detected cracks are masked and indexed, and their centers are mapped from pixel to real-world coordinates using GeoJSON polygon corners.

### Example Workflow

**Input image with grid overlay:**

<p align="center">
<img src="https://github.com/Vlasenko2006/Crack_Finder/blob/main/Crack_samples/Crack_unclassified1.png" alt="Input image" width="50%">
</p>

**Example wall coordinates (GeoJSON format):**
```
geojson_polygon = {
    "type": "Polygon",
    "coordinates": [[
        [9.73114, 53.565534, 1],   # upper-left corner
        [9.73160, 53.566134, 1],   # upper-right corner
        [9.73160, 53.566134, 0],   # lower-right corner
        [9.73114, 53.565534, 0],   # lower-left corner
        [9.73114, 53.565534, 1],   # close polygon
    ]]
}
```

**Detected and masked cracks:**

<p align="center">
<img src="https://github.com/Vlasenko2006/Crack_Finder/blob/main/Crack_samples/Cracks_classified1.png" alt="Classified cracks" width="50%">
</p>

**Patch center coordinates of detected cracks:**
```
Patch_id 4  coordinates (lat,lon,alt): (53.565931, 9.731444366666667, 0.8744444444444445)
Patch_id 7  coordinates (lat,lon,alt): (53.5655905, 9.731183316666666, 0.6222222222222222)
Patch_id 14 coordinates (lat,lon,alt): (53.565704, 9.731270333333333, 0.37)
Patch_id 15 coordinates (lat,lon,alt): (53.5658175, 9.73135735, 0.37)
Patch_id 16 coordinates (lat,lon,alt): (53.565931, 9.731444366666667, 0.37)
Patch_id 17 coordinates (lat,lon,alt): (53.5660445, 9.731531383333333, 0.37)
Patch_id 18 coordinates (lat,lon,alt): (53.566077, 9.7315563, 0.37)
Patch_id 21 coordinates (lat,lon,alt): (53.5658175, 9.73135735, 0.1266666666666667)
Patch_id 24 coordinates (lat,lon,alt): (53.566077, 9.7315563, 0.1266666666666667)
```

**Demo: Crack detection in video**

<p align="center">
  <a href="https://youtu.be/4QStHUmI6J4" target="_blank">
    <img src="https://img.youtube.com/vi/4QStHUmI6J4/0.jpg" alt="Crack detection video" width="480">
    <br>
    <strong>Watch Crack detection video</strong>
  </a>
</p>

---

## Code Structure

- **Crack_NN.py** &mdash; Fine-tunes a pretrained `ResNet-18` for crack detection. Employs transfer learning, checkpointing, and tqdm progress visualization.
- **Split_dataset.py** &mdash; Splits the Kaggle dataset into train and validation sets.
- **Large_crack_check.py** &mdash; Grids the input image, masks cracks, and outputs patch centers with geographic coordinates.
- **Large_crack_check_video.py** &mdash; Processes video frame-by-frame, masking cracks in real time.

---

## Features

- **Transfer Learning:** Utilizes a pretrained ResNet-18 backbone for fast convergence.
- **Custom Dataset Support:** Expects images organized into `Positive` and `Negative` folders.
- **Automatic Checkpointing:** Saves model and optimizer states after each epoch.
- **Progress Visualization:** Uses tqdm for real-time training feedback.
- **Geo-referencing:** Converts detected crack locations to geographical coordinates using user-provided polygons.

---

## Installation

Create and activate a Python virtual environment (optional, but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required dependencies:

```bash
pip install torch torchvision tqdm
```

---

## Dataset Structure

Organize your dataset as follows:

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

Use the provided script or your own method to split your dataset into train/val sets.

---

## Training the Model

Run the training script:

```bash
python Crack_NN.py
```

Model checkpoints will be saved in the `checkpoints` directory after each epoch, and the final model weights will be saved as `crack_resnet18.pth`.

---

## Key Parameters

- **Batch Size:** 32 (modifiable)
- **Image Size:** 227x227 pixels
- **Epochs:** 3 (adjust as needed)
- **Learning Rate:** 0.001 (Adam optimizer)

---

## Inference Usage

After training, load the model for inference or fine-tuning:

```python
import torch
from torchvision import models

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('crack_resnet18.pth'))
model.eval()
```

---

## Notes & Tips

- For best results, ensure your validation set is representative of your use case.
- Adjust folder paths and hyperparameters as needed for your dataset.
- Geographic conversion requires correct GeoJSON polygon input.

---

## License

This project is licensed under the MIT License.
