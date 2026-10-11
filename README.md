# Quad Remesh for Blender

Field-aligned quad remeshing for Blender 4.2+ (tested on 5.2). It turns any mesh into
evenly sized, all-quad topology that follows sharp edges and optional hand-drawn
flow guides, with as few poles ("stars") as possible.

Pure Python + NumPy: nothing to compile, installs as a single zip.

## Install

1. Build the zip (or use a prebuilt one from `dist/`):
   ```sh
   blender --command extension build --source-dir quad_remesh --output-dir dist
   ```
2. In Blender: **Edit > Preferences > Get Extensions > ⌄ > Install from Disk…** and
   pick `dist/quad_remesh-0.1.0.zip`.
3. Open the 3D View sidebar (**N**) and select the **Quad Remesh** tab.

## Use

| Control | What it does |
|---|---|
| **Density** | How many quads fit across the mesh. The estimated quad count and edge length are shown underneath. |
| **Show Checker** | Overlays a checker grid on the active mesh. Each square is the size of one output quad. |
| **Sharp Edges › By Angle** | Edges sharper than the angle become hard features. The new edge flow runs along them, and an edge loop lands exactly on each one. The check is made at quad scale, so scan noise is ignored. |
| **Follow Marked Sharp Edges** | Treats edges you marked with *Edge › Mark Sharp* as features. |
| **Draw Guides** | Click it, then **LMB-drag** on the surface to draw strokes the flow should follow. **Shift+LMB-drag** erases and **Enter/RMB/Esc** finishes. Strokes are stored in a curve object (`QR_Guides_<name>`), so you can edit them too. Any curve object can be used as guides. |
| **Strength** | How strongly the guides steer the flow (1 = hard constraint). |
| **Pure Quads** | Guarantees zero triangles and n-gons. |
| **Reduce Poles** | Globally straightens the grid (removes lattice dislocations) and runs topology clean-up. |
| **Refine Thin Parts** | Uses smaller quads where the part is thinner than a quad (ears, fins, cloth). Adds a few poles at the size transitions, so it's off by default. |
| **Relax** | Iterations that even out quad shape and size. |
| **Remesh** | Runs in the background with a progress bar (Esc cancels). Creates `<name>_Quad`, and running it again updates that same object. Feature edges are marked sharp on the result. |

Modifiers are applied: the evaluated mesh is what gets remeshed. The source object is never
changed. Use **Toggle Original** to compare.

## How it works

1. **Prepare:** merge duplicates and refine long edges. Detect sharp, marked and boundary feature curves (scale-aware, so noise is rejected). Smooth normals for the field.
2. **Orientation field:** an extrinsic 4-RoSy cross field (Instant Meshes), solved coarse-to-fine on a multiresolution hierarchy. Features are hard constraints and guides are soft or hard constraints.
3. **Position field:** a 4-PoSy lattice whose spacing comes from Density. Lattice lines are pinned onto feature curves.
4. **Global quantization:** integer lattice dislocations (which would become valence 3/5 pairs) are cancelled along fold-free paths. This uses capacity limits and orientation checks, in the style of QuadriFlow. Quads are then rebuilt combinatorially from the corrected lattice, and positions are re-solved by conjugate gradients.
5. **Clean-up:** fill holes, pair odd faces, split n-gons, and do one midpoint subdivision for pure quads. Then edge rotation and diagonal collapse, shape-matching relaxation, reprojection (component- and normal-aware), and untangling.
6. **Safety net:** if the quantized result looks worse (stars, flips, holes, size outliers), the classic extraction is also run and the better one is kept.

## Results (from `tests/compare.py`)

| Mesh | Quads | Poles (v3/v5/v6+) |
|---|---|---|
| Cube | 600 | 8 / 0 / 0 (perfect grid) |
| Torus | 1728 | 0 / 0 / 0 (perfect rings) |
| Capped cylinder | 570 | 8 / 0 / 0 |
| Noisy scan sphere | 1504 | 8 / 0 / 0 |
| UV sphere | 1660 | 13 / 5 / 0 |
| Beveled box | 932 | 10 / 2 / 0 |

A closed surface always needs some poles (a sphere-like shape needs at least eight valence-3
vertices). The tool keeps everything beyond that to a minimum.

## Known limitations

- Parts thinner than a quad (Suzanne's ears at low density) can collapse or flip. Raise
  Density or enable **Refine Thin Parts**.
- Separate shells that intersect each other (e.g. eyeballs pushed into a head) are remeshed
  independently but can look rough where they meet.
- Speed: about 1–4 s for typical inputs, about 10–20 s for 60k-triangle inputs at high density.
- Low-poly inputs where many faceted edges exceed the angle threshold: lower the angle, or
  turn off *By Angle* and mark the sharp edges you want.

## Development

```sh
# headless test suite (13 tests: core quality + add-on integration)
blender -b --factory-startup --python tests/run_tests.py

# compare settings across meshes
blender -b --factory-startup --python tests/compare.py -- "sphere:40,torus:40" "quantize=False|True"

# render a result (irregular vertices marked, flipped faces in red)
blender -b --factory-startup --python tests/render_result.py -- torus 40 /tmp/torus.png
```

`quad_remesh/core/` contains no `bpy` imports (only NumPy and `mathutils`), so the algorithm can
be tested without the add-on.
