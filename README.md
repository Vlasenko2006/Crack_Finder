# Concrete Crack Detection with ResNet-18

## Intro 

This is a ResNet-18 neural network trained for crack detection. Give it a foto of an inspected object, or even a video, and it will spotify cracks, delivering their locations (as geojson geographical coordinates). The neural network is trained within 3 epochs on a dataset of crfacks taken from [Kaggle repository](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection), where each sample is a 227x227 image of cracked or uncracked concrete.  


The detection script uses this neural network to detect cracks. It accepts an image of a given construction, put a grid on it where each cell has the same size as the size of an image from trained dataset (227x227 pixels). It classifies each cell for having or not having cracks and masks the cells classified as cracked. See example an below:

### Input image after putting mask on it
<p align="center">
<img src="https://github.com/Vlasenko2006/Crack_Finder/blob/main/Crack_samples/Crack_unclassified1.png" alt="input image" width="50%">
</p>

### Wall coordinates (not real) in Geojson format: 
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


### Spotted and classified cracks

<p align="center">
<img src="https://github.com/Vlasenko2006/Crack_Finder/blob/main/Crack_samples/Cracks_classified1.png" alt="input image" width="50%">
</p>

### Spotted crack centers coordinates: 
```
Patch_id 4 coordinates (lat,lon,alt): (53.565931, 9.731444366666667, 0.8744444444444445)
Patch_id 7 coordinates (lat,lon,alt): (53.5655905, 9.731183316666666, 0.6222222222222222)
Patch_id 14 coordinates (lat,lon,alt): (53.565704, 9.731270333333333, 0.37)
Patch_id 15 coordinates (lat,lon,alt): (53.5658175, 9.73135735, 0.37)
Patch_id 16 coordinates (lat,lon,alt): (53.565931, 9.731444366666667, 0.37)
Patch_id 17 coordinates (lat,lon,alt): (53.5660445, 9.731531383333333, 0.37)
Patch_id 18 coordinates (lat,lon,alt): (53.566077, 9.7315563, 0.37)
Patch_id 21 coordinates (lat,lon,alt): (53.5658175, 9.73135735, 0.1266666666666667)
Patch_id 24 coordinates (lat,lon,alt): (53.566077, 9.7315563, 0.1266666666666667)
```

Here is a short video example of spotting cracks on a fly: 

<p align="center">
  <a href="https://youtu.be/4QStHUmI6J4" target="_blank">
    <img src="https://img.youtube.com/vi/4QStHUmI6J4/0.jpg" alt="Crack detection video" width="480">
    <br>
    <strong>Watch Crack detection video</strong>
  </a>
</p>


## Code structure

- **Crack_NN.py** fine tunes pretraied `ResNet-18` for crack detection. The model uses transfer learning, checkpointing, and a tqdm progress bar for efficient training on your own dataset.
- **Split_dataset.py** splits Kaggle dataset into train and validation sets.
- **Large_crack_check.py** Grids the input image, masks the cracks with patches, identfying patches' geographical centers.
- **Large_crack_check_video.py** Grids the input video, masks the cracks with patches on a fly.    




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
