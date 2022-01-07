[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bjnLYm4FF-kAvEnqReYRbFG4-1mmQMgJ?usp=sharing)

# Birds 325 Model

A Vision Transformer model that is capable of predicting 325 species of birds. Model is trained on this [dataset](https://www.kaggle.com/gpiosenka/100-bird-species).

Training code can be found [here](https://www.kaggle.com/sharansmenon/birds-325-pytorch/notebook). This model is the `deit_tiny_patch16_224` model. It requires an input size of 224x224.

More information about DeIT can be found [here](https://github.com/facebookresearch/deit).

Dependencies:

- [`torch`](https://www.pytorch.org/)
- [`timm`](https://github.com/rwightman/pytorch-image-models)

Install dependencies with `pip install torch timm`


## Load Model

```python
import torch
import timm # Required dependency for loading model
model = torch.hub.load("SharanSMenon/birds-325-model", "birds_325_deit_tiny_patch16_224")
```

## Load Classes

```python
import json
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/SharanSMenon/birds-325-model/main/classes.json"
response = urlopen(URL)
classes = json.loads(response.read())
```

## Inference Code

`PIL` and `torchvision` are required dependencies for inference.

```python
from PIL import Image
import torchvision.transforms as T

test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

image = Image.open("painted-bunting.jpg")
transformed = test_transform(image)
batch = transformed.unsqueeze(0)
with torch.no_grad():
  output = model(batch)
prediction = classes[output.argmax(dim=1).item()]
print(prediction)
```