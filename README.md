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
| **Symmetry** + **X / Y / Z** | Remeshes one half and mirrors it, so the result is exactly symmetric with an edge loop on the mirror plane. The plane goes through the object origin, perpendicular to the chosen local axis (like the Mirror modifier), and the + side is the one remeshed. While Symmetry is on, **Draw Guides mirrors each stroke** across the axis as you draw (a fainter preview shows the copy, and erasing removes both). A stroke that reaches the mirror line **joins its mirror image there**, just like a loop closes: draw half a mouth ring from the centre line and back and you get the whole symmetric ring; draw a brow line out from the centre and it runs across both sides. Anything drawn past the line is trimmed. Strokes along the centre line are kept as drawn. |
| **Draw Guides** | Click it, then **LMB-drag** on the surface to draw a stroke; the result gets an **edge loop along every stroke**. End a stroke back where it started (the preview closes) to make a **closed ring**, e.g. around an eye or the mouth. **Shift+LMB-drag** erases and **Enter/RMB/Esc** finishes. Strokes are stored in a curve object (`QR_Guides_<name>`), so you can edit them too; cyclic splines are rings. Any curve object can be used as guides. |
| **Rings Out / In** | Extra **full edge loops** around (and inside) every closed guide, one lattice cell apart (two quad rows with Pure Quads): the concentric rings you want around eyes and mouths. Default 1 outside, 0 inside. A ring is left out where it would run into an opening, another guide, or get too small; when the rings of two guides would collide (eyes close together), both are left out, so the result stays even. |
| **Joining strokes** | A stroke that starts or ends at the end of another open stroke snaps onto it and they become one stroke (ending on both ends of the same stroke closes it into a ring). Hold **Ctrl** while drawing to keep a stroke exactly as drawn: no snapping, no joining at the mirror line and no auto-closing. |
| **Stroke Smoothing** | Takes the hand wobble out of newly drawn strokes (wiggles up to ~1.5 quads long at 1). Rings keep their size and open strokes keep their ends. 0 keeps strokes exactly as drawn. Default 0.5. |
| **Influence** | How far around each guide the flow bends to follow it. The loop on the guide is always kept; 0 = just the loop, 1 = the flow runs parallel to it up to ~4 quads away (concentric rings). |
| **Pure Quads** | Guarantees zero triangles and n-gons. |
| **Reduce Poles** | Globally straightens the grid (removes lattice dislocations) and runs topology clean-up. |
| **Refine Thin Parts** | Separate pieces thinner than a quad (caps, shells, fins, glass walls) are remeshed at their own smaller, even quad size so both sides and the rim stay clean. Adds faces on those pieces only. On by default. |
| **Detail** | Keeps detail smaller than a quad (small fillets, ridges, embossing) by using gradually smaller quads there, down to 1/4 size. 0 keeps every quad the same size (the default, and best for organic / character meshes). Detail too fine to keep (e.g. knurling far below the quad size) is smoothed over instead of producing broken geometry. |
| **Smoothing** | Passes that even out quad size and squareness after the quads are built. Vertices stay on the original surface, on sharp edges and on guide loops while they move. 0 = raw output; higher = more even quads, at the cost of softening very small surface detail slightly. |
| **Remesh** | Runs in the background with a progress bar (Esc cancels). Creates `<name>_Quad`, and running it again updates that same object. Feature edges are marked sharp on the result. |

Modifiers are applied: the evaluated mesh is what gets remeshed. The source object is never
changed. Use **Toggle Original** to compare.

## How it works

0. **Input preparation (CAD-friendly):** a non-destructive clean-up welds only identical vertices and never deletes triangles. Very dense inputs are simplified to an even "field mesh" the solvers run on: Blender's decimator, then evening out (split long / safely collapse short / flip), with Blender's voxel remesher as a fallback for closed parts the decimator leaves too uneven. Projection, sharp edges and detail always come from the original mesh.

1. **Prepare:** merge duplicates and refine long edges. Detect sharp, marked and boundary feature curves (scale-aware, so noise is rejected). Smooth normals for the field.
2. **Guides:** closed strokes first get their concentric rings: offset copies marched across the surface one lattice cell at a time, which are added as strokes of their own. Each stroke is dropped onto the surface and a chain of field-mesh edges along it is moved onto it. That chain is a line constraint like a sharp edge, but it does not split normal smoothing and is not marked sharp on the result.
3. **Orientation field:** an extrinsic 4-RoSy cross field (Instant Meshes), solved coarse-to-fine on a multiresolution hierarchy. Features and guide chains are hard constraints; a band around each guide (width set by Influence) is a soft one.
4. **Position field:** a 4-PoSy lattice whose spacing comes from Density. Lattice lines are pinned onto feature curves and guide strokes.
5. **Global quantization:** integer lattice dislocations (which would become valence 3/5 pairs) are cancelled along fold-free paths. This uses capacity limits and orientation checks, in the style of QuadriFlow. Quads are then rebuilt combinatorially from the corrected lattice, and positions are re-solved by conjugate gradients.
6. **Clean-up:** fill holes and seal every opening that is not an opening of the input (gaps are measured against the input's real boundary, of any size), pair odd faces, split n-gons, and do one midpoint subdivision for pure quads. Each guide is then traced as a continuous edge path through the result and snapped onto its stroke. Then edge rotation and diagonal collapse, shape-matching relaxation, reprojection (component- and normal-aware), and untangling.
7. **Safety net:** if the quantized result looks worse (holes, broken guide loops, stars, flips, size outliers), the classic extraction is also run and the better one is kept. Holes away from any input opening rule a result out.

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

- Thin pieces get one smaller quad size for the whole piece, so their face count can rise a
  lot (a cap shell 1/5 of a quad thick gets ~40x more quads). Thin spots on otherwise thick
  pieces (Suzanne's ears) are not refined locally; raise Density there.
- Detail finer than 1/4 of a quad (knurling, threads) is smoothed over.
- Separate shells that intersect each other (e.g. eyeballs pushed into a head) are remeshed
  independently but can look rough where they meet.
- Each extra ring pushes the poles further out, so expect a few more of them around ringed areas.
- A ring guide forces some poles inside it (a quad ring cannot close without them), and a ring
  needs roughly 16+ quads around to come out clean; for small features, raise Density.
- Speed: about 1–5 s for typical parts; a 250k-triangle sculpt takes ~13 s at density 50. A 1M-triangle bottle takes ~6 s at density 20, or
  ~60 s when its thin glass wall triggers refinement to ~17k quads.
- Low-poly inputs where many faceted edges exceed the angle threshold: lower the angle, or
  turn off *By Angle* and mark the sharp edges you want.

## Development

```sh
# headless test suite (16 tests: core quality, symmetry, add-on integration)
blender -b --factory-startup --python tests/run_tests.py

# compare settings across meshes
blender -b --factory-startup --python tests/compare.py -- "sphere:40,torus:40" "quantize=False|True"

# render a result (irregular vertices marked, flipped faces in red)
blender -b --factory-startup --python tests/render_result.py -- torus 40 /tmp/torus.png
```

`quad_remesh/core/` contains no `bpy` imports (only NumPy and `mathutils`), so the algorithm can
be tested without the add-on.

## License

Quad Remesh is free software, licensed under the
[GNU General Public License v3.0 or later](LICENSE) (`GPL-3.0-or-later`), the same
license family as Blender. You may use, study, share and modify it. If you distribute it
or a modified version, you must do so under the same license and make the source available.
