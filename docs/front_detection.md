# Front Detection

## Phase 1: numerical core only

Phase 1 provides pure NumPy/SciPy functions for detecting and tracking a
concentration front on an existing structured x-z grid. It does not perform
file access, provide a CLI, or create research outputs. The threshold is a
fixed absolute concentration value; the core does not normalize concentration.

The sequence API is:

```python
track_concentration_front(
    time,
    Xi,
    Zi,
    C_frames,
    threshold=...,
    min_component_pixels=...,
    bottom_rows=...,
    max_front_jump=...,
    connectivity=8,
)
```

The required shapes are:

```text
time:     (nt,)
Xi:       (nz, nx)
Zi:       (nz, nx)
C_frames: (nt, nz, nx)
```

Coordinates and time are converted to floating-point NumPy arrays. Time must
be finite, nonempty, unique, and strictly increasing. Non-finite concentration
cells are permitted and excluded from detection.

## Detection and spatial filtering

For each frame, the active mask is defined strictly as:

```python
np.isfinite(Xi) & np.isfinite(Zi) & np.isfinite(C_grid) & (C_grid > threshold)
```

A concentration equal to the threshold is therefore inactive. The active mask
is labeled with `scipy.ndimage.label`; 8-connectivity using the full 3-by-3
neighborhood is the default, and 4-connectivity is also supported.

The detector reports every component's pixel count, x and z extents, grid
mask, and bottom-contact flag. Bottom contact is based on coordinate values,
not array row order: the unique finite z levels are sorted, and the highest of
the lowest `bottom_rows` levels defines the cutoff. `bottom_rows` is clipped to
the available z levels. A component is spatially valid only if it contacts
this bottom region and has at least `min_component_pixels` active cells.

Floating components and undersized bottom fragments are rejected. The front
position is the maximum x coordinate of the selected component.

## Initialization and temporal tracking

The first frame with a spatially valid component initializes tracking. Its
component is selected by:

1. greatest pixel count;
2. greatest maximum x;
3. smallest component label.

Earlier failed frames remain missing. With one successful detection, the next
predicted front is the previous front. With two or more successful detections,
the last two successes define a constant-velocity extrapolation:

```text
velocity = (last_x - previous_x) / (last_time - previous_time)
predicted_x = last_x + velocity * (current_time - last_time)
```

This supports non-uniform time spacing. Failed sequence frames are not added
to the successful history.

`max_front_jump` is an absolute x-distance tolerance around the predicted
position, not a speed. Positive infinity disables distance rejection. Among
components within the tolerance, selection uses:

1. greatest pixel overlap with the most recently selected component;
2. greatest component pixel count;
3. smallest absolute distance from the prediction;
4. greatest maximum x;
5. smallest component label.

Overlap therefore has priority over component area and proximity. The signed
tracking error is:

```python
selected_x_front - predicted_x
```

## Status and failure behavior

Every frame receives exactly one status:

```text
selected_initial
selected_tracked
no_threshold_component
no_valid_spatial_candidate
no_valid_temporal_candidate
```

The three failure statuses distinguish an empty threshold mask, rejection by
component size or bottom contact, and rejection by the temporal jump limit.
Failed frames have a NaN front and tracking error, label `-1`, zero selected
pixels and overlap, NaN selected extents, and false selected bottom contact.
A prediction remains finite when successful history exists. The previous
front is never copied into a failed frame, and tracking can recover later
using the last successful history.

## Phase boundaries

`front_simple.dat` is not used by the detector. Phase 1 also does not include:

```text
Nek5000 loading
slice extraction
interpolation
morphological cleanup
contour tracing
CSV writing
plotting
comparison with front_simple.dat
```

## Phase 2: N7 processing and reference-comparison workflow

Phase 2 connects the numerical core to Nek5000 snapshots. N7 is the current
primary case. The workflow:

1. discovers exact `GC0.fNNNNN` files and sorts their five-digit indices;
2. extracts the nearest-plane y-midspan concentration slice from each file;
3. creates one fixed x-z grid from the first selected snapshot;
4. interpolates every concentration slice to that same grid;
5. calls the unchanged Phase 1 tracker;
6. compares successful detections with `front_simple.dat`;
7. writes diagnostic CSV files and reference-comparison figures.

The default fixed grid is 500 by 200 points, and concentration interpolation
is linear. Interpolation NaNs remain NaN and are excluded by the Phase 1
finite mask. No intermediate slice or grid files are written.

The automatic front is the primary result. `front_simple.dat` is used only as
an external reference for post-hoc comparison and is not treated as ground
truth. It is read after automatic detection and never guides threshold
selection, component labeling, filtering, temporal prediction, candidate
ranking, front selection, or success/failure decisions. Automatic and
teacher-provided front workflows therefore coexist.

At overlapping successful times, the reported method-to-method difference is:

```text
difference = x_auto - x_front_simple
```

Agreement or disagreement describes a trend comparison between independently
defined curves; it is not an automatic-front accuracy score.

### Provisional defaults

The CLI exposes these current detection defaults:

```text
case: N7
threshold: 0.01
min_component_pixels: 50
bottom_rows: 3
max_front_jump: 0.5
connectivity: 8
grid: 500 x 200
slice_mode: nearest_plane
interpolation_method: linear
```

These are provisional starting values, not validated final research
parameters. In particular, `max_front_jump` is a distance around the predicted
x position, not a velocity.

For a small smoke test, restrict the file range and grid:

```bash
PYENV_VERSION=research312 python scripts/16_detect_front_from_concentration.py \
  --case N7 \
  --start-index 1 \
  --end-index 2 \
  --nx 100 \
  --nz 40 \
  --overwrite
```

Run the full configured N7 sequence with:

```bash
PYENV_VERSION=research312 python scripts/16_detect_front_from_concentration.py \
  --case N7 \
  --overwrite
```

By default, outputs are written to
`results_root/front_detection/N7` using these exact names:

```text
N7_detected_front_timeseries.csv
N7_front_detection_comparison.csv
N7_front_detection_summary.csv
N7_front_detection_overlay.png
N7_front_detection_difference.png
```

The timeseries retains failed frames as NaN detections. Status meanings are:

- `selected_initial`: first spatially valid component selected;
- `selected_tracked`: component accepted using temporal prediction and ranking;
- `no_threshold_component`: no finite cell exceeded the fixed threshold;
- `no_valid_spatial_candidate`: components existed, but all failed minimum-size
  or bottom-contact filtering;
- `no_valid_temporal_candidate`: spatial candidates existed, but all were
  farther than `max_front_jump` from the prediction.

### Parameter tuning

Inspect the timeseries status counts, finite interpolation fraction,
method-to-method differences, and both plots before changing parameters.
Adjust `threshold` for the absolute concentration level of the coherent
current, and `min_component_pixels` to reject isolated fragments at the chosen
grid resolution. Use `bottom_rows` to accommodate the interpolated
bottom-contact band without admitting detached objects. Tune `max_front_jump`
last, using an x-distance large enough for plausible prediction uncertainty
while still rejecting downstream jumps. Reassess parameters if the grid
resolution, time sampling, slice mode, or interpolation method changes.

Threshold, component-size, bottom-contact, temporal-jump, and grid sensitivity
must be assessed from physical consistency and robustness, not by minimizing
the difference from `front_simple.dat`. The automatic method is more
explicitly defined, reproducible, spatially filtered, and temporally
consistent; agreement with the external reference alone does not establish
greater accuracy.
