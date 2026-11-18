# NNJvt input storage trial

Apply `integration.diff` to the tree with `git apply --check` first. The only target is
`provider/small_jvt.cxx`. The trial define is `JETS_NNJVT_REUSE_INPUTS`. Compiling the
patched file with the flag absent preserves the original prediction block. The
source file revision and SHA-256 are frozen beside the patch.

The release-matched `FastGraph<double>` takes a const reference to `NodeVec<double>`.
The implementation constructs a local `LazySource` and computes the result
before its return. The input and preprocessor references do not escape. The
trial leaves FastGraph, preprocessing, Eigen layer code and the score cast and
comparison intact. The four float-valued features still reach the same Eigen
comma initializer in the same order. A lease remains active through compute,
and the returned values own their storage.

There is one lazily allocated outer vector with exactly one Eigen vector of
four doubles per participating thread. The bounded coefficient storage is 32
bytes, plus the outer element, the thread-local container and flag, and
allocator overhead. It does not scale with events, jets, graphs or systematic
states. Nested eligible calls take the usual local construction. The slot is
marked occupied before an allocation or feature read. A failed allocation
restores the slot and rethrows. Scope exit also releases it on unwind. All four
coefficients are replaced before use, so the storage can serve different graph
objects without depending on the previous contents. The one-time allocation can
still fail. The optimization removes later allocations from the ordinary path.

Eligibility starts after the original output buffer, vertex and feature checks,
float conversions, RpT, maximum-pT comparison and cut-map binning. Direct C
also excludes whole calls already covered by the nominal-reuse option. The
portable combined 2,000-event baseline had 2,329,002 update rows. That is a call
count upper bound for this trial on that run, before the above gates or nominal
reuse. The 3.85% combined baseline sample share in the three named allocator
leaves covers more than construction of input wrappers. Inference and activation
allocations remain. No final run call rate or whole-workflow gain is inferred.

## Focused fixture

From the authoritative WSL checkout, choose a fresh output and the approved
canonical payload. Run these commands under the assigned build and fixture
slots. The fixture uses one CPU and one execution thread. It exercises nested leases, scratch ownership and reset after exceptions in that thread.

```bash
variant=provenance/provider-patches/nnjvt-inputs
output="$PWD/results/diagnostics/nnjvt-inputs-fixture-001"
taskset -c "$BUILD_CPU" bash "$variant/build-fixture.sh" "$output" "$APPROVED_JETS_JSON"
LD_LIBRARY_PATH="$(cat "$output/library-path.txt")" taskset -c "$FIXTURE_CPU" \
  "$output/nnjvt-inputs-fixture" "$output/baseline.so" "$output/candidate.so" \
  "$output/nnjvt-update.json" "$output/cuts-and-network.json"
```

The build copies only the pinned baseline into an isolated directory, applies
the patch there, and compiles independent provider-only shared libraries
against the same SDK, release headers and static portable lwtnn 2.14.1. Both
use the same provider floating-point flags. Loading uses separate CorrectionLib
registries. Payload preparation extracts the existing update correction and
checks the embedded graph and FixedEffPt cuts against the precise pinned
CVMFS assets. The build records the model, hashes, compiler flags and linkage.

The fixture checks all four output bits, exceptions and partial output writes,
the rounding mode, and sticky exception-flag equality. It visits the real cut
edges and their binary32 neighbors, the pT update cutoff, negative and signed-zero
coordinates, subnormals and nonfinite inputs. Raw plugin calls cover gate and
error precedence. Test-only probes hold the real lease through a nested
evaluation, verify the same-thread fallback and separate thread ownership,
and check slot release during unwinding. An eight-row public batch covers
borrowed reversed features and a broadcast input. The probes stay inside the
fixture library. Model execution always uses the real graph.

After the fixture, qualify a dedicated provider differing only by this define
from the chosen same-stream reference. Keep the wheel, script behavior,
selected defaults, data and evaluator settings fixed as appropriate for this
paired comparison. Start with small-R and combined and use large-R as a
negative control if measuring the full provider grid. Require the complete stock
object and histogram gates and the campaign timing spread criteria before
adopting a result. A focused fixture alone does not qualify a timing variant.
