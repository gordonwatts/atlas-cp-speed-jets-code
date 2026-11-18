# Original-stream MT19937 prefix on cache misses

This is a source-only isolated trial. The active provider, wheel and benchmark
controller do not use it. The patch changes the portable provider only when
`JETS_RNG_PREFIX`, `JETS_PORTABLE` and `JETS_RNG_CACHE` are defined together.
The fixture stages both sets of headers under a fresh output directory.

## What the packed profile leaves to remove

Bundle-005 used the packed payload and the installed visible combined full
workflow with nominal NNJVT reuse. Its exact full-stock histogram comparison
passed. The native software CPU-clock capture excludes the first pass and
covers the next nine 20,000-event passes. It records 133.583 s elapsed in that
window, a 131.62 s measured worker CPU floor, and 132.96 s of decoded sample
periods. Use that sample denominator to interpret the executing leaves below.

| Executing engine work | Sample count | CPU-clock seconds | Share |
| --- | ---: | ---: | ---: |
| MT seed and its modulo helper, disjoint leaves | 878 | 8.78 | 6.603% |
| Full-state twist | 329 | 3.29 | 2.474% |
| Total target | 1,207 | 12.07 | 9.078% |

The 9.078% share is a ceiling on eliminating those **observed leaves**, or
1.341 sampled CPU seconds per profiled pass. The entire resolution draw path,
including work this trial retains, is 10.221% inclusive. These shares do not
predict elapsed speedup, do not include initialization and do not bound the
unresolved profiler symbols. The earlier 21.83% unoptimized seed share is not
the relevant estimate. On the separate 1 MiB cache instrumentation, small-R
made 1,352,612 misses on 5,321,414 requests, compared with 14,084 misses on
187,380 large-R requests. All four recorded negative restarts were small-R.
Those are distinct earlier diagnostics, not counters from bundle-005. They
support testing small and combined first and treating large as lower payoff.

With all tail promotions excluded, the arithmetic removes 195ºw^~)Þw225 of 623
initialization updates and 608ºw^~)Þw623 of 624 twist words, depending on how many
words a normal requests. Applying those fractions to each of the sampled
leaf groups gives a work-proportional illustration of roughly 4.54.9% of
the packed processing window before new cursor, continuation and code-layout
costs. These fractions are not supported as time coefficients. A focused
measurement would determine whether the mechanism warrants a full pair.

Machine receipts within the authoritative checkout:

- `results/diagnostics/packed-profile-decoded-001/self.tsv`, SHA-256
  `dfc2e30b786de23c2cf0b7447113b98631e7c8effcfa6ac584d999be18ef78cf`.
- `results/diagnostics/packed-profile-decoded-001/inclusive.tsv`, SHA-256
  `7582cd8f289cc75a1a33dae1cc3027bc5611d18ef91ce9a09655a188c94349e3`.
- `results/validation/packed-loading-001/native-profile/recording.json`, SHA-256
  `67e818f8c6e0e197dfcf034cfe6beb0e636c62c4833fb7a49777eefc8811c11b`.
- `results/packaging/bundle-005/bundle.json`, SHA-256
  `f8ef15de7ac3e92cdad458dea14edf258a7246616fc55159088fb8c39e3968cf`.
- `results/benchmarks/paired-stream-001/count-{small,large}.json`.

## Why the raw words agree

The recipe in `provider/jet_smearing.h` derives a signed integral seed from
the absolute value of the original float phi. The standard MC multiplier is
200,000 and pseudo-data uses 123,000. Truncation to zero takes the configured
positive fallbacks 45583453 and 34545654 respectively. The exporter freezes
the default user seed at zero, meaning use these recipe settings. The prefix
never changes seed derivation. It enters after the existing positive-seed,
finite-width and round-to-nearest guards, and only on a cache miss.
Direct callers with a positive wider seed retain the original modulo-2^32
narrowing even when the resulting initial word is zero. The true zero seed
stays excluded from the portable generator.

For the scalar 32-bit initialization, `s[0]` is that narrowed seed and
`s[j] = 1812433253 × (s[j1] xor (s[j1] >> 30)) + j` modulo 2^32.
The first twist uses the original `s[i]`, `s[i+1]` and `s[i+397]` for
`0 <= i < 227`, before any of those inputs has been overwritten. It joins
the top bit of `s[i]` with the lower 31 bits of `s[i+1]`, shifts right,
uses the least significant bit to select the standard twist constant, and
XORs with `s[i+397]`. It applies the unchanged four tempering operations.
The implementation advances two unsigned 32-bit cursors and creates no
state array. It advances the high cursor to `s[397]` once, then delays
further recurrence steps until the next raw word is requested. A prefix of
`k` words therefore does `397 + k + (kºw^~)Þv1)` recurrence operations and
`k` twists. A fixed limit of 16 is safely below the first-loop boundary.

The Gaussian uses the same source operations in both implementations,
including the conversion of a nonzero raw word by multiplication by 2^-32.
Raw zeros still occupy stream positions and are skipped before uniform
conversion. On the seventeenth requested raw word, the existing standard
engine is initialized from the same narrowed seed, skips exactly 16 raw
words and returns the next raw word to the **same** Gaussian evaluation.
Its previous floating-point operations are neither replayed nor discarded.
It then supplies the rest of that one-shot Gaussian for as long as needed.
The prefix promises only the first standard Gaussian. Every scalar fallback
or rejected negative factor restarts from the original seed in the existing
scalar loop. This retains width-zero and nonfinite behavior, cache hit
ordering and the source's rejection sequence. The ROOT-backed implementation
keeps its current calls.

The exact algorithms are visible in the pinned libstdc++ 15.2
`bits/random.tcc` seed, first twist loop and tempering, and the ROOT
sources below. The release linked by the fixture remains the authority
for the uniform-to-Gaussian calculation. [Pinned ROOT TRandom3](https://github.com/root-project/root/blob/d4c0003542952bec9c4edc86c419784a32875e9d/math/mathcore/src/TRandom3.cxx) · [ROOT 6.40 Gaussian source](https://root.cern/doc/v640/TRandom_8cxx_source.html)

## Focused verification recipe

Run only once build and fixture slots are authorized. These commands do not
apply the patch to the shared checkout.

```bash
variant=provenance/provider-patches/rng-prefix
output="$PWD/results/diagnostics/rng-prefix-fixture-001"
taskset -c "$BUILD_CPU" bash "$variant/build-fixture.sh" "$output"
export LD_LIBRARY_PATH="$(cat "$output/library-path.txt")"
taskset -c "$FIXTURE_CPU" "$output/raw-and-gaussian" > "$output/raw-and-gaussian.log" 2>&1
taskset -c "$FIXTURE_CPU" "$output/resolution-baseline" > "$output/resolution-baseline.log" 2>&1
taskset -c "$FIXTURE_CPU" "$output/resolution-candidate" > "$output/resolution-candidate.log" 2>&1
diff -u "$output/resolution-baseline.log" "$output/resolution-candidate.log"
```

The word fixture checks published-sequence answers at seed 5489 and compares
to standard MT19937 below and across the 1, 16 and 227-word boundaries,
including a further twist. It forces raw zero skips and a one-word Gaussian
continuation. It covers representative seeds across the configured MC and
pseudo-data domains, both configured fallbacks, 32-bit wrap, wider positive
seeds, four rounding modes and preexisting sticky exceptions. Its target
rounding mode checks the Gaussian against release ROOT. The existing
resolution fixture compared in separate processes checks widths, negative
factor restart, applicability and lookup order, special seeds and identical
diagnostic counts. Before accepting production, also close exact object and
histogram gates for the affected provider and compare matched original-stream
endpoints. Use the same flags, cache size, data, corrections, script and
histogram work. A macro-off build of the patched shared uniform adapter can
check for code-layout effects separately from the already accepted baseline.

No build, fixture, model execution, profiling or timing was performed for
this source-only package.
