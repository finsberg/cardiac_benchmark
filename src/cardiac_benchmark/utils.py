import logging
from typing import Any
from typing import Dict
from typing import Optional

import dolfin

logger = logging.getLogger(__name__)


def _update_parameters(
    _par: Dict[str, Any],
    par: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if par is None:
        par = {}
    for key, value in par.items():
        if key not in _par:
            logger.warning(f"Invalid key {key}")
            continue

        if isinstance(_par[key], dolfin.Constant):
            _par[key].assign(value)
        else:
            _par[key] = value
    return _par
