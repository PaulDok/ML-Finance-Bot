import logging

from src.core import utils
from src.core.config import Config, get_config


def main() -> None:
    """
    Main entry point for project
    Run from command line -> should parse config and run in specified mode (e.g. launch Streamlit server)
    """
    logging.getLogger().setLevel("INFO")
    config = main_launch()
    # launch actual process
    # TBD


def main_launch() -> Config:
    """
    May be used to get config in a Jupyter Notebook and do something specific
    """
    # Parse config
    config = get_config()
    # # Print config params (for debug)
    # for key, value in config.dict().items():
    #     print(key, value, type(value))
    utils.init_logger(config)

    return config


if __name__ == "__main__":
    main()
