# ML-Finance-Bot
ML Finance trading bot for OTUS cource

# Environment setup
Dependencies are managed using poetry

In order to set up the environment run:

```
conda create -n <env_name> python=3.9
conda activate <env_name>

# To work in jupyter notebooks
conda install ipykernel

# Linter for development - is not necessary for replication
conda install black
conda install isort

# Poetry for more complex dependency management
conda install poetry

# Install project dependencies
cd app
poetry install
```