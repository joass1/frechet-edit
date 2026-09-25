# Releasing

The package is built and checked here, but **publishing is a manual step taken
by the maintainer**. Nothing in this repository uploads anything, and no CI job
has a PyPI token. That is deliberate: `docs/limitations.md` records that
publication is a decision, not a build artefact.

## Before you publish anything

A release says "this is fit to depend on". Check that it is:

- [ ] `pytest -q -m "not slow and not realdata"` green
- [ ] `pytest -q -m slow` green
- [ ] `ruff check src tests experiments benchmarks examples webapp` clean
- [ ] `mypy` clean
- [ ] CI green on all nine matrix jobs for the commit you are releasing
- [ ] `CHANGELOG.md` has a dated section for this version
- [ ] the version matches in **three** places: `pyproject.toml`,
      `src/frechet_edit/__init__.py`, `CITATION.cff`
- [ ] `docs/reviews/STATUS.md` still describes reality, including what has NOT
      been reviewed

The last one matters more than the rest. This package is alpha and nothing in it
has had an independent review; a release must not quietly imply otherwise.

## Build and check

```bash
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*
```

Then install the built wheel in a throwaway environment and run it from
**outside** the checkout, so nothing resolves from the source tree by accident:

```bash
python -m venv /tmp/fresh
/tmp/fresh/bin/python -m pip install dist/frechet_edit-*.whl
cd /tmp
/tmp/fresh/bin/python -c "
import numpy as np, frechet_edit
ref, obs = np.array([[0.],[1.],[0.]]), np.array([[0.]])
r = frechet_edit.discrete_edit_distance(ref, obs, 0.4, operations='insert', return_witness=True)
assert r.cost == 2, r
assert frechet_edit.verify_witness(ref, obs, r).ok
print('ok', frechet_edit.__version__)
"
```

## Publish

Upload to TestPyPI first and install from it. A broken description or a missing
file is cheap to find there and permanent on PyPI.

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple frechet-edit
```

Then the real thing:

```bash
python -m twine upload dist/*
```

Authenticate with an API token: username `__token__`, password the token
itself. Use a **project-scoped** token once the project exists. Never commit a
token, and never paste one into a shell that records history.

## After

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

Then open a GitHub release pointing at the changelog section.

## A release cannot be undone

A version number on PyPI is permanent. `twine upload` of a given version can
happen exactly once; deleting a release does not free the number. If something
is wrong, the fix is a new patch version, never a re-upload. This is the reason
for the TestPyPI step above.
