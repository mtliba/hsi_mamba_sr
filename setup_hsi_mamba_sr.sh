#!/bin/bash
# setup_hsi_mamba_sr.sh
# Create project structure for HSI-Mamba-SR

PROJECT="hsi-mamba-sr"

mkdir -p $PROJECT/hsimamba/models
mkdir -p $PROJECT/hsimamba/data
mkdir -p $PROJECT/hsimamba/utils
mkdir -p $PROJECT/hsimamba/config
mkdir -p $PROJECT/scripts
mkdir -p $PROJECT/tests

# core init
touch $PROJECT/hsimamba/__init__.py

# models
touch $PROJECT/hsimamba/models/ops.py
touch $PROJECT/hsimamba/models/blocks.py
touch $PROJECT/hsimamba/models/groups.py
touch $PROJECT/hsimamba/models/backbone.py

# data
touch $PROJECT/hsimamba/data/datasets.py
touch $PROJECT/hsimamba/data/transforms.py

# misc core
touch $PROJECT/hsimamba/losses.py
touch $PROJECT/hsimamba/metrics.py

# utils
touch $PROJECT/hsimamba/utils/logger.py
touch $PROJECT/hsimamba/utils/checkpoint.py
touch $PROJECT/hsimamba/utils/seed.py
touch $PROJECT/hsimamba/utils/prof.py

# train/eval
touch $PROJECT/hsimamba/train.py
touch $PROJECT/hsimamba/eval.py

# config
touch $PROJECT/hsimamba/config/base.yaml
touch $PROJECT/hsimamba/config/cave_x2.yaml
touch $PROJECT/hsimamba/config/cave_x4.yaml
touch $PROJECT/hsimamba/config/icvl_x2.yaml
touch $PROJECT/hsimamba/config/icvl_x4.yaml

# scripts
touch $PROJECT/scripts/prepare_cave.py
touch $PROJECT/scripts/run_cave_x4.sh

# tests
touch $PROJECT/tests/test_shapes.py
touch $PROJECT/tests/test_block_grad.py

# root files
touch $PROJECT/README.md
cat <<EOF > $PROJECT/requirements.txt
torch>=2.0
mamba-ssm>=2.0
numpy
tqdm
pyyaml
scipy
h5py
EOF

echo "Project skeleton created at $PROJECT/"
