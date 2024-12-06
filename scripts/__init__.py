import sys
import logging
from pathlib import Path

sys.path.append(str(Path(f"{__file__}/../../").resolve()))
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
