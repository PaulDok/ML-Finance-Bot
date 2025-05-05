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

# Local Database setup
Project uses a local SQLite database, saved as ./app/data/cache.db

However, it's in .gitignore to limit project volume (and should be updated by most recent data in runtime anyway), so to avoid errors in Homework notebooks it should be populated first, in case this is a fresh project clone.

The easiest way to do it by far is to use interactive UI:
- [launch Streamlit UI](#run-interactive-ui-in-browser)
- expand the first section, "Update local data cache controls"
- alter controls if necessary and click "Update data in local cache" button
- after a while SQLite database will be populated with raw and preprocessed data

Under the hood it calls
```
utils.update_tickers_data
```
with parameters from inputs, so you may choose to do that manually as well

# Run interactive UI in browser
In order to run a Streamlit page after environment is set up, run:

```
cd app
python main.py
```
This will parse launch parameters from dev_env and launch Streamlit UI in your browser
