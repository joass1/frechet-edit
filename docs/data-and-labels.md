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

Additional items, and their current resolution status:

* ~~There is a version mismatch between the published GeoLife v1.2 user guide
  and the archive currently offered for download.~~ **RESOLVED.** A GeoLife
  archive has since been obtained locally and verified. It is
  `Geolife Trajectories 1.3` and it ships its own authoritative
  `User Guide-1.3.pdf`, so the v1.2 guide is no longer the reference for it.
  Release history in that guide: v1.3 released 2012/08/01. Contents verified:
  182 user folders, 18670 `.plt` trajectory files, 69 `labels.txt` files.
* **Licence verified from the shipped v1.3 guide, section 7**, the Microsoft
  Research License Agreement (Non-Commercial Use Only). Quoting the scope of
  rights directly:

  > You may use this Software for any non-commercial purpose, subject to the
  > restrictions in this MSR-LA. Some purposes which can be non-commercial are
  > teaching, academic research, public demonstrations and personal
  > experimentation. **You may not distribute this Software or any derivative
  > works in any form.**

  So the assumption this document was already written against is confirmed
  rather than overturned: non-commercial academic use is permitted, and
  redistribution of the data **or of derivative works** is not. The rule in
  section 1 stands unchanged and is now sourced rather than inferred.
* Availability of a free download does not establish a redistribution right.

The local archive lives under `data/raw/`, which is gitignored. Nothing derived
from it may be committed, and no fixture, test or example may depend on a path
inside it.

## 2. The harder problem is labels, not access

Even with the data in hand, neither dataset supplies the label this project
needs.

* GeoLife documents **transportation-mode** labels. A transportation mode is not
  a route identity. Confirmed against the local v1.3 archive: `labels.txt` has
  columns `Start Time`, `End Time`, `Transportation Mode`, with values such as
  `bus` and `train`, and only 69 of 182 users carry a labels file at all. This
  is the label that exists, not the label this project needs, so obtaining the
  data does **not** unblock evidence level C.
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
