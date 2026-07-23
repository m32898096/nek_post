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

Phase 2 will provide N7 file processing and validation against the relevant
research outputs.
