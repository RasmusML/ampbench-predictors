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
    entry_point="amPEPpy.predict:predict",
    repo_path="./plugins/thirdparty/ampeppy",
    repo_url="https://github.com/szczurek-lab/seqme-thirdparty",
    branch="ampeppy",
    python_bin="/opt/anaconda3/envs/ampeppy_env/bin/python",
)

model(sequences=["SEQVENCE"])
```
