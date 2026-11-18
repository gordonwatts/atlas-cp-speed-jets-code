# Adjacent sign-pair trial

Apply the patch after the active measurements. The base revision is in `base-revision.txt`. The patch keeps the existing CorrectionLib node, inputs, eight output values, state order, and default behavior. The optional C lease retains exact typed row snapshots for one adjacent pair within a collection chunk. The lease holds at most 65,536 rows. Unsupported states and lease conditions use the scalar calculation. Ordinary and pseudo-data pairs are separate.

A captured aggregate precedes the model's signed parameter. The ordinary and pseudo-data application paths still evaluate status, multiply by the configured sign, run the existing stochastic logic, and write the original coordinates and decorations. Groups with more than one scale keep the scalar path.

## Validation after the reserved slots open

From the authoritative checkout, retain the previously accepted native options in both arms. This example uses the same RNG cache in both arms. If the accepted checkpoint uses different compiled controls, replace the matching flags consistently.

```bash
bundle=provenance/provider-patches/unsigned-sign-pairs
git apply --check "$bundle/integration.diff"
git apply "$bundle/integration.diff"

trial=unsigned-sign-pair-001
core=core-precision-002
lwtnn=portable-lwtnn-001
taskset -c 5 bash tools/build-portable.sh "$trial" "$core" "$lwtnn" rng-cache unsigned-sign-pairs
test "$(cat "results/build/$trial/exit-code")" = 0

sdk="$PWD/results/build/$core/venv/lib/python3.14/site-packages/correctionlib"
py="$PWD/results/build/$core/venv/bin/python"
provider="$PWD/results/build/$trial/libjets.so"
gcc_lib=/cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/lib64
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD
export LD_LIBRARY_PATH="$sdk/lib:$gcc_lib"
taskset -c 3 "$py" -m tools.test_unsigned_sign_pairs "$provider" "results/validation/$trial-fixture.json"

template=results/diagnostics/joint-plans-001/rng-size.json
plan="results/diagnostics/$trial-plan.json"
"$py" "$bundle/prepare-plan.py" "$template" "$provider" "$plan" 1
taskset -c 3 bash tools/screen_trials.sh "$plan" "results/benchmarks/$trial"
```

The screen checks both arms against stock object values and then their complete uninstrumented histograms. Round one gives a diagnostic. The kept trial needs a fresh plan with four or more rounds, the same SDK and payload in both arms, and full per-object comparisons for both settings. Use the released A observer and the existing `tools/compare_columnar.py` command with `--events 20000 --chunk 128`, once without the flag and once with `--reuse-uncertainty-sign-pairs`. Also compare the compiled trial with its flag off against the pre-trial accepted provider. This measures overhead from the dormant path.

The focused fixture calls CorrectionLib using synthetic models with declared signs, arbitrary names, a validity mask, a b-jet component, a resolution component, and a multi-scale group. It checks exact scalar and vector output bits, order, pseudo-data availability, mixed model identity, unsupported and retained subsets, borrowed input mutation, wrong row counts, exceptional and unscoped calls, and lease teardown. It requires nonzero native reuse for every eligible pair and checks fallback-only cases by their hit counts.

The source-derived count receipt is `results/diagnostics/unsigned-group-discovery-001/summary.json`. Adjacent signs would remove half of the 27,238,604 accepted component-value calls. The measured baseline samples assign 9.665% of its processing window to component-value work. Those figures describe eligibility and measured baseline attribution.
