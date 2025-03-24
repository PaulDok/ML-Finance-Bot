# ML-Finance-Bot
ML Finance trading bot for OTUS cource

# Environment setup
Dependencies are managed using poetry

In order to set up the environment run:

```
conda create -n <env_name> python=3.10
conda activate <env_name>

# Poetry for more complex dependency management
conda install poetry

# Install project dependencies
cd app
poetry install

# TA-Lib (TODO: can it be done via poetry?)
conda install conda-forge::ta-lib
```



# Run interactive UI in browser
In order to run a Streamlit page after environment is set up, run:

```
cd app
python main.py
```
This will parse launch parameters from dev_env and launch Streamlit UI in your browser
