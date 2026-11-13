# Central dependency compatibility

The declared interpreter range remains **standard CPython 3.11 through 3.14**
(`>=3.11,<3.15`). This document records published-package evidence, not a
successful installation, native import matrix, or notebook run. The separately
identified target-resolution checks below do not establish those runtime claims.

## Coordinated direct-pin choices

| Distribution | Selected version | Published evidence and reason |
|---|---|---|
| azure-functions | 1.24.0 | [Release metadata](https://pypi.org/pypi/azure-functions/1.24.0/json): `Requires-Python >=3.7`, non-yanked universal Python 3 wheel. The previous [2.3.0 release](https://pypi.org/pypi/azure-functions/2.3.0/json) requires `>=3.13`, excluding two declared minors. The manifest is bounded below 2 to avoid reintroducing that floor. |
| numpy | 2.3.5 | [Release metadata](https://pypi.org/pypi/numpy/2.3.5/json) and [published files](https://pypi.org/project/numpy/2.3.5/#files): `Requires-Python >=3.11`, with standard `cp311`, `cp312`, `cp313`, and `cp314` wheels. The previous 2.2.6 release has no `cp314` wheels in the inspected inventory. |
| opencv-python | 4.13.0.90 | [Release metadata](https://pypi.org/pypi/opencv-python/4.13.0.90/json): on Python 3.9+, requires `numpy>=2`; published `cp37-abi3` wheels include Windows AMD64, Linux x86-64, and macOS variants. The previous [4.12.0.88 release](https://pypi.org/pypi/opencv-python/4.12.0.88/json) requires `numpy>=2,<2.3` there, so upgrading NumPy alone would conflict. |

Functions 1.24.0 is a local library selection, not a claim about the Python
versions supported by an Azure Functions hosted runtime. Import/API checks
against retained consumers remain required; no hosted deployment or runtime
compatibility has been demonstrated by this metadata review.

## All direct Python floors were inspected

The offline [publisher-metadata snapshot](shared/tests/data/dependency_metadata.json)
records the exact version, version-specific official PyPI JSON URL, and
`Requires-Python` field for **all 35 direct pins**, including the test extra.
The [regression tests](shared/tests/test_dependency_compatibility.py) require
the snapshot to match the lock, check every recorded Python floor against all
four declared minors, and check the published OpenCV/NumPy constraint together.
They neither contact PyPI nor perform resolution. Refresh the snapshot from
publisher evidence when changing a pin; editing a snapshot is not validation.

Jupyter 1.1.1 publishes no `Requires-Python` value. It is a universal-wheel
metapackage whose dependencies must be resolved for the target environment;
the missing field is recorded as unknown, not converted into a support claim.

Actual CPython 3.14 wheel inventories were also inspected for the retained
compiled direct packages: aiohttp 3.13.3, matplotlib 3.10.7, pandas 2.3.3,
Pillow 12.3.0, PyArrow 24.0.0, PyYAML 6.0.3, scikit-learn 1.7.2, and
tiktoken 0.12.0. Their version-specific publisher URLs are in the snapshot.
This checks more than classifiers: for example, matplotlib's classifiers stop
at 3.13 even though its release includes actual `cp314` wheels.

## Platform and transitive boundaries

- Wheel examples were checked for Windows AMD64 and Linux x86-64; this is not
  a full cross-platform installation matrix. Several selected Linux wheels
  require glibc 2.28. Older Linux systems are not established as compatible.
- macOS wheel availability and minimum OS versions vary by package and
  architecture. No complete macOS environment was installed in this review.
- The inspected OpenCV release has **no native Windows ARM64 wheel**. Native
  ARM64 installation is not covered by the evidence above.
- Free-threaded CPython, PyPy, source compilation, and every transitive native
  dependency are not established by direct-package wheel filenames.
- This is a direct lock, not a complete transitive or hashed lock. Among the
  transitive dependencies still requiring resolver/import checks are
  Pydantic Core, SciPy, cryptography, and the OpenAI client brought in by Azure
  AI Projects. Do not equate a compatible Python floor with a working install.
- Local package compatibility does not establish model access, hosted-runtime
  support, quota, or paid notebook execution success.

## Maintainer validation and CI coverage

### Target-resolution checks, 2026-10-22

The full direct lock and its transitives were resolved with `uv pip compile`,
binary wheels only, against the repository's existing Microsoft package proxy.
No package was installed, no project index setting changed, and no certificate
validation was disabled. These are cross-target resolutions from Windows, not
fresh interpreters running each target environment.

| Standard CPython target | Windows x86-64 | Linux x86-64, glibc 2.28 |
|---|---|---|
| 3.11 | Resolved, 170 pins | Blocked by feed HTTP 500 fetching matplotlib metadata |
| 3.12 | Resolved, 169 pins | Resolved, 170 pins |
| 3.13 | Resolved, 169 pins | Blocked by feed HTTP 500 fetching aiohttp metadata |
| 3.14 | Resolved, 169 pins | Blocked by feed HTTP 500 fetching tiktoken metadata |

The initial public-PyPI resolution attempt stopped at a TLS handshake failure on
the artifact host; it did not establish a dependency conflict. The Linux feed
failures likewise leave those targets unverified. Azure Functions is absent
from the disposable regression-test environment, so it supplies no native
import result for the changed pin. Repeat clean installs and native imports in
the intended environments before claiming runtime support.

Before publishing an updated lock, use fresh environments on every declared
minor to perform full resolver dry-runs, including transitives, then clean
installs, package-consistency checks, native import checks, and the focused
offline packaging/cookbook tests. Resolve on the actual OS/architecture; a
cross-platform wheel query alone does not evaluate every environment marker.
Retain the resolver reports and distinguish them from runtime test results.

The [published documentation workflow at the inspected source revision](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/.github/workflows/documentation-checks.yml)
covers Windows and Ubuntu with Python **3.11 and 3.13** only. It installs a
separate documentation-test dependency set, not this central lock, and its
explicit test selection does not provide packaging or new notice-test
coverage. Thus it is not central-installer evidence for 3.11-3.14; 3.12/3.14
and central-lock installer coverage need an explicit CI follow-up. No installer
result is asserted by these resolution checks. PR3's managed offline CI now
selects the packaging and publisher-metadata regressions as well; it still
does not install this notebook lock or validate the native interpreter matrix.