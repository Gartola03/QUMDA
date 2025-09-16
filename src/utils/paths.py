from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Sub-paths
DATA_DIR = PROJECT_ROOT / "data"
GRAPH_DIR = DATA_DIR / "graphs"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TEST_DIR = OUTPUT_DIR / "test"
RESULTS_DIR = OUTPUT_DIR / "results"


OPT_DIR = OUTPUT_DIR / "results" / "classicalSolution" / "OptimalSolutions"

CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

CSC_DIR = OUTPUT_DIR/ "csv"



