from __future__ import annotations
from typing import Sequence
from pathlib import Path
import xml.etree.ElementTree as Et
import os


def write_pvd(
    directory: os.PathLike | str,
    name: str,
    file_dirs: Sequence[Path],
    times: Sequence[float],
) -> Path:
    r"""
    Write a PVD file for a collection of VTK files.
    :param directory: Directory to write the PVD file to.
    :param name: Name of the PVD file.
    :param file_dirs: File directories of the VTU files to refer to.
    :param times: Time history corresponding to each VTU file.
    :return: Path of the written PVD file.
    """
    dirpath = Path(directory)
    dirpath.mkdir(parents=True, exist_ok=True)

    if len(file_dirs) == 0:
        raise ValueError("filenames must be a non-empty sequence")
    if len(file_dirs) != len(times):
        raise ValueError("filenames and times must have the same length")

    # Build XML tree
    vtk_file = Et.Element(
        "VTKFile",
        {
            "type": "Collection",
            "version": "0.1",
            "byte_order": "LittleEndian",
        },
    )
    collection = Et.SubElement(vtk_file, "Collection")

    if not name.endswith(".pvd"):
        name += ".pvd"

    pvd_path = dirpath.joinpath(name)

    for filedir, t in zip(file_dirs, times):
        rel_filedir = filedir.relative_to(dirpath)
        Et.SubElement(
            collection,
            "DataSet",
            {
                "timestep": f"{float(t):.04f}",
                "group": "",
                "part": "0",
                "file": str(rel_filedir.as_posix()),
            },
        )

    Et.indent(vtk_file, space="  ")
    tree = Et.ElementTree(vtk_file)
    tree.write(pvd_path, encoding="utf-8", xml_declaration=True)

    return pvd_path
