from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Sub-paths
DATA_DIR = PROJECT_ROOT / "data"
GRAPH_DIR = DATA_DIR / "graphs"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

AER_SIM_PLOT_DIR = PLOT_DIR / "AerSimulator"
FAKE_PLOT_DIR = PLOT_DIR / "FakeBacked"
REAL_PLOT_DIR = PLOT_DIR / "RealHardware"


AER_SIM_DIR = OUTPUT_DIR / "results" / "AerSimulator"
FAKE_DIR = OUTPUT_DIR / "results" / "FakeBacked"
REAL_DIR = OUTPUT_DIR / "results" / "RealHardware"

CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"



