from __future__ import annotations
from typing import Sequence
from pathlib import Path
import xml.etree.ElementTree as Et
from os import PathLike

def write_pvd(directory: PathLike,
              name: str,
              filedirs: Sequence[Path],
              times: Sequence[float],
              ) -> Path:

    dirpath = Path(directory)
    dirpath.mkdir(parents=True, exist_ok=True)

    if len(filedirs) == 0:
        raise ValueError("filenames must be a non-empty sequence")
    if len(filedirs) != len(times):
        raise ValueError("filenames and times must have the same length")

    # Build XML tree
    vtkfile = Et.Element("VTKFile", {
        "type": "Collection",
        "version": "0.1",
        "byte_order": "LittleEndian",
    })
    collection = Et.SubElement(vtkfile, "Collection")

    if not name.endswith(".pvd"):
        name += ".pvd"

    pvd_path = dirpath.joinpath(name)

    for filedir, t in zip(filedirs, times):
        rel_filedir = filedir.relative_to(dirpath)
        Et.SubElement(collection, "DataSet", {
            "timestep": f"{float(t):.04f}",
            "group": "",
            "part": "0",
            "file": str(rel_filedir.as_posix()),
        })

    Et.indent(vtkfile, space="  ")
    tree = Et.ElementTree(vtkfile)
    tree.write(pvd_path, encoding="utf-8", xml_declaration=True)

    return pvd_path

