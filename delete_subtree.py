import numpy as np
import neurom as nm
import morphio
from morphio import MissingParentError
from typing import Optional

import numpy as np
import morphio
from morphio import MissingParentError

def neurite_type_name(morphio_type):
    mapping = {
        0: "UNDEFINED",
        1: "SOMA",
        2: "AXON",
        3: "BASAL_DENDRITE",
        4: "APICAL_DENDRITE",
        5: "GLIA",
    }
    return mapping.get(int(morphio_type), str(morphio_type))


def remove_subtree(
    input_swc,
    output_swc,
    *,
    section_id=None,
    root_xyz=None,
    match_tol=1e-6,
    recursive=True,
    fix_unifurcations=True,
):
    """
    Remove a subtree from an SWC/H5/ASC morphology and write to a new file.

    Args:
        input_swc (str): path to input morphology file.
        output_swc (str): path to output morphology file.
        section_id (int): MorphIO section id to delete (preferred).
        root_xyz (tuple): (x, y, z) to match a section endpoint if section_id not given.
        match_tol (float): distance tolerance for root_xyz matching.
        recursive (bool): if True, delete entire subtree under root.
        fix_unifurcations (bool): if True, call remove_unifurcations() before writing.

    Returns:
        int: section id that was deleted.
    """
    morph = morphio.Morphology(input_swc).as_mutable()

    # Helper: euclidean distance
    def d3(a, b):
        return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))

    target_id = None

    if section_id is not None:
        target_id = int(section_id)
        root_sec = morph.section(target_id)
    elif root_xyz is not None:
        best = None
        best_dist = float("inf")
        for s in morph.sections:
            pts = np.asarray(s.points, dtype=float)
            start = pts[0, :3]
            end   = pts[-1, :3]
            dist = min(d3(root_xyz, start), d3(root_xyz, end))
            if dist <= match_tol and dist < best_dist:
                best = s
                best_dist = dist
        if best is None:
            raise ValueError(f"No section endpoint within {match_tol} of {root_xyz}")
        root_sec = best
        target_id = int(best.id)
    else:
        raise ValueError("Must specify section_id or root_xyz")

    morph.delete_section(root_sec, recursive=recursive)

    if fix_unifurcations and hasattr(morph, "remove_unifurcations"):
        morph.remove_unifurcations()

    morph.write(output_swc)
    return target_id
