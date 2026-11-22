"""Publication figure font and locked class names.

Times New Roman is not installed. STIX General, shipped with Matplotlib, is the
Times-like face used for every results figure.

Class names in every figure and table are fixed:
ODD uses Non-ODD and ODD. DICCD uses Non-DICCD and DICCD.
Do not rename these to Absent, No ODD, Control, or case.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

CLASS_LABELS = {
    "ODD": ["Non-ODD", "ODD"],
    "DICCD": ["Non-DICCD", "DICCD"],
}

_STIX = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
_REGISTERED = False


def use_times_style() -> None:
    """Set the Times-like face. Call this before creating a figure."""
    global _REGISTERED
    if not _REGISTERED:
        for name in (
            "STIXGeneral.ttf",
            "STIXGeneralBol.ttf",
            "STIXGeneralItalic.ttf",
            "STIXGeneralBolIta.ttf",
        ):
            font_manager.fontManager.addfont(str(_STIX / name))
        _REGISTERED = True
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
