# Data availability, rights and labels

Summary: **no real trajectory data is present in this repository, none was
downloaded, and none may be committed here.** Every fixture, example and
benchmark curve is synthetic and generated from a seed.

## 1. Why there is no data here

The two candidate datasets for this work are GeoLife and T-Drive. Both are
distributed by Microsoft Research under terms that permit non-commercial
research use but do NOT permit redistribution of the data or of derivative
works. That has two consequences which are treated as binding:

1. Raw trajectories may not be committed to this repository.
2. **Transformed** fixtures - projected, cropped, resampled, corrupted or
   otherwise derived - may not be committed either, because they are derivative
   works.

Therefore the public demos, tests and examples are synthetic. This is a licence
constraint, not a preference, and it is the reason evidence level B is BLOCKED
in `docs/experiment-protocol.md`.

Additional unresolved items recorded rather than assumed:

* There is a version mismatch between the published GeoLife v1.2 user guide
  and the archive currently offered for download. Before any use, the
  actual archive's licence and version must be verified; the older guide is not
  authoritative for a different archive.
* Availability of a free download does not establish a redistribution right.

## 2. The harder problem is labels, not access

Even with the data in hand, neither dataset supplies the label this project
needs.

* GeoLife documents **transportation-mode** labels. A transportation mode is not
  a route identity.
* T-Drive documents taxi id, timestamp, longitude and latitude. A **taxi id is
  not a route identity** either, and the dataset's reported average sampling
  interval of about 177 seconds and 623 metres makes it a sparse-sampling
  stress test rather than a source of known paths between observations.

So "which template is the correct one for this query" is not available from
either dataset without independent annotation. That is why evidence level C
requires blinded annotators and a written route-equivalence rubric, and why it
is BLOCKED here rather than approximated.

**Pseudo-labels derived from Frechet or FED scores are forbidden.** Using the
measure under evaluation to generate the labels it is evaluated against is
circular and would invalidate every number produced.

## 3. If real data is provisioned later

The following must all hold before a level B or C claim is made.

1. **Provenance.** Record the exact URL, download date, file size and checksum.
   A download script must fail loudly on an HTML error page, a truncated
   archive or a checksum mismatch, rather than proceeding with a corrupt file.
2. **Terms.** Re-read the licence attached to the archive actually obtained and
   log it. Do not rely on a guide for a different version.
3. **Storage.** Raw and interim data live under `data/raw` and `data/interim`,
   which are git-ignored. They are never packaged into a wheel or an sdist.
4. **Projection.** Latitude and longitude must be projected to a local metric
   CRS before reaching this library. The core is Euclidean: it never applies
   haversine distance and never treats degrees as metres. Record the chosen CRS
   and the spatial extent it is valid for, and test the forward and inverse
   transforms.
5. **Segmentation.** Split trips with a preregistered time-gap policy. Report
   counts and reasons for duplicate timestamps and invalid coordinates.
6. **Outliers.** Do not remove suspected outliers from the raw arm before
   evaluating an outlier-robust method; doing so would delete the very signal
   under test. Keep a raw arm always.
7. **Splits.** Manifests are frozen before any corruption or parameter choice.
   No source trajectory, overlapping segment, corruption derivative or
   duplicate may straddle validation and test. Test this with an explicit
   disjointness check, not by inspection.
8. **Bootstrap units.** Resample by independent source trajectory, route or
   person - not by individual correlated corrupted queries. Many pairwise
   comparisons are not many independent samples.
9. **Marker.** Tests touching provisioned data use the `realdata` pytest
   marker. They never run in CI, and a missing dataset must SKIP loudly rather
   than let an absent-data path masquerade as a passing real-data gate.

## 4. Current status

| requirement | status |
|---|---|
| licensed local dataset | NOT PRESENT |
| download performed | NO |
| parser exercised on real data | NO |
| route-identity labels | NOT AVAILABLE |
| independent annotators | NOT AVAILABLE |
| level B evidence | BLOCKED |
| level C evidence | BLOCKED |

This table is the honest state. It is not a to-do list that has been partially
completed and rounded up.
