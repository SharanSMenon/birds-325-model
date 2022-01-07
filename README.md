# Birds 325 Model

Dependencies:

- [`torch`](https://www.pytorch.org/)
- [`timm`](https://github.com/rwightman/pytorch-image-models)


## Load Model

```python
import torch
import timm # Required dependency for loading model
model = torch.hub.load("SharanSMenon/birds-325-model", "birds_325_deit_tiny_patch16_224")
```

### Load Classes

```python
import json
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/SharanSMenon/birds-325-model/main/classes.json"
response = urlopen(url)
classes = json.loads(response.read())
```