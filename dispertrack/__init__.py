from pathlib import Path


home_path = Path.home()
config_path = Path.home() / '.dispertrack'
if not config_path.exists():
    config_path.mkdir(parents=True)


# from .start_waterfall_analysis import start_analysis
