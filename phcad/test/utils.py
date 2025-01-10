import logging
import json

logger = logging.getLogger(__name__)


def check_results(savepath, check_aupro=False):
    try:
        with open(savepath, "r") as f:
            results = json.loads(f.read())
        auroc_completed = "auroc" in results
        aupro_completed = "aupro" in results if check_aupro else True
        if auroc_completed and aupro_completed:
            logger.info(
                f"Results already saved to {savepath}, delete if needed to run again"
            )
            return True
    except Exception:
        logger.info(f"Results failed to load from {savepath}, regenerating them")
        return False
