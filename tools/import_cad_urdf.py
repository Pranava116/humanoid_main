"""Turn the CAD URDF export into the stack's URDF (settings in config/stack/cad.yaml).

Run again after every CAD re-export or joint-limit change in robot.yaml, then
regenerate the sim scene (robot.run does that automatically when the URDF is newer).

Examples:
  python tools/import_cad_urdf.py
  python tools/import_cad_urdf.py --source "models/Assembly001 (3).urdf" --out models/humanoid_v3.urdf
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.config import load_config  # noqa: E402
from robot.sim.cad_import import convert, specs_from_config  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", help="config directory (default config/stack)")
    p.add_argument("--source", help="CAD URDF (default cad.yaml source)")
    p.add_argument("--out", help="output URDF (default cad.yaml output)")
    p.add_argument("--meshdir", help="folder with the .stl meshes (default cad.yaml meshdir)")
    args = p.parse_args(argv)

    cfg = load_config(args.config_dir)
    specs, cad = specs_from_config(cfg)
    src = args.source or cfg.path(str(cad["source"]))
    out = args.out or cfg.path(str(cad["output"]))
    meshdir = args.meshdir or cfg.path(str(cad.get("meshdir", "models/meshes")))
    ax = cad.get("axes", {})
    root = convert(src, out, meshdir, specs,
                   axes=(str(ax.get("forward", "+x")), str(ax.get("left", "+y")), str(ax.get("up", "+z"))),
                   root_name=str(cad.get("root_link", "torso")), effort=float(cad.get("effort", 1.0)),
                   name=os.path.splitext(os.path.basename(out))[0])
    n_links = len(root.findall("link"))
    mass = sum(float(m.get("value")) for m in root.iter("mass"))
    print(f"wrote {out}: {len(specs)} joints, {n_links} links, {mass:.3f} kg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
