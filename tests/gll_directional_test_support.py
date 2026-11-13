"""Synthetic inputs shared by directional-GLL workflow and CLI tests.

Each call creates fresh arrays; importing this helper performs no file I/O.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nek_post.gll import gll_nodes


def make_directional_data(
    *,
    bad_concentration_shape: bool = False,
    velocity_components: tuple[float, float, float] | None = None,
) -> SimpleNamespace:
    nodes = gll_nodes(3)
    z, y, x = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    concentration = np.ones_like(x)
    if bad_concentration_shape:
        concentration = concentration[0]
    velocity = (
        (x + 1.0, y + 2.0, z + 3.0)
        if velocity_components is None
        else tuple(np.full_like(x, component) for component in velocity_components)
    )
    return SimpleNamespace(
        elem=(
            SimpleNamespace(
                pos=(x, y, z),
                temp=(concentration,),
                vel=velocity,
                pres=(x - y + z,),
            ),
        )
    )
