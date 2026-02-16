# OmegAMP-classifiers

### Installation

Create a new virtual enviroment, e.g., `conda` environment:

```
conda create -n omegamp-env python=3.11
```

Define the entry-point.

```python
from seqme.models import ThirdPartyModel

model = ThirdPartyModel(
    entry_point="omegamp.predict:predict",
    repo_path="./plugins/thirdparty/omegamp",
    repo_url="https://github.com/RasmusML/ampbench-predictors",
    #branch="ampeppy",
    python_bin="/opt/anaconda3/envs/omegamp-env/bin/python",
)

model(sequences=["SEQVENCE"])
```
