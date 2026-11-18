# Bounded variable-axis lookup trial

This patch adds a single native choice, `variable-axis-index`. Both builds keep `fast-variable-flow` enabled to isolate bin search from the flow clamp. No payload format, calibration equation, interpolation operation or state dispatch changes.

An eligible axis has 24 through 65,535 variable bins and a finite reciprocal scaling range. Its 65 two-byte indices narrow a global lower bound to one coarse value interval. The query must be strictly above the preceding edge and at most the upper sentinel edge. Sorted edges make those checks sufficient to rule out every omitted element. If either check fails, the operation uses the original global search. Fixed and smaller variable axes keep their existing paths. The original comparison chooses flow bins for NaN, infinity and the final edge. Equal edges and both signs of zero use the existing first-equal convention.

## Archived evidence

`results/diagnostics/packed-profile-decoded-001` recorded 13,296 samples with `joint-baseline-001` underneath pristine bundle 005. Logical leaf attribution accounted for all 1,003 joint double lower-bound samples. The flow clamp accounts for 479 of them. The remaining 524 are a whole-window upper bound of 3.94% for one joint axis-lookup change, before axis eligibility, index overhead and sampling error. They split into 230 1D/interpolated, 150 bilinear neighbor centers, 90 other bilinear, 52 direct/unexpanded and two trilinear samples. Another 53 such leaves outside that owner were attributed to GSC, forward-fold JMR and JMS. The portable EtaJES axes are all fixed.

The complete packed payload `results/export/eta-jes-packed-001/jets.json` has SHA-256 `1aed4df05b75b208ec23dc61c3de9d04e76b5f90cc561407d2e20ed89a0a4cc7`. Its uncertainty tables hold 83 small-R and 156 large-R variable-axis instances. Only one distinct variable geometry per radius is regular. At bin midpoints eligible small-R geometries average 10.69 original binary steps and 5.09 estimated narrowed steps. The analogous large-R depths are 7.27 and 2.25. These figures weight each stored eligible bin's midpoint once per serialized axis and exclude the constant cost of index arithmetic and guards.

The plan loads 190 eligible uncertainty axes and generates 404 eligible 1D projection axes. On x86-64 each axis object adds 32 bytes of fields with the trial enabled. The 130-byte allocation for eligible axes gives approximately 96.6 KiB persistent joint uncertainty storage before allocator overhead. Treating all further serialized axes as live gives approximately 147 KiB across the provider. Construction does 64 short binary lookups per eligible axis. The measurement must include provider setup and the complete original-file workflow.

The archived fixture includes the duplicated first X edge at 200 in large-R JMS FF stat7, the duplicated last Y edge at 700 in its JMS validity histogram and the duplicated first Y edge at zero in independent large-R JMR component 9. The synthetic fixture additionally covers repeated interior edges, signed zeros, exact seams and adjacent ULPs, tiny ranges, and a coarse estimate rounded across a boundary where the narrowed range would omit the first equal edge.

## Isolated build and validation

Use an isolated checkout at the revision containing this bundle. The checkout needs the pinned `source/correctionlib/rapidjson`, precision SDK under `results/build/core-precision-002` and lwtnn under `results/build/portable-lwtnn-001` as in the campaign. Use a fresh destination and the authorized build CPU.

```bash
dir=provenance/provider-patches/variable-axis-index
git apply --check "$dir/integration.diff"
git apply "$dir/integration.diff"
taskset -c 5 bash tools/build-portable.sh axis-index-off core-precision-002 portable-lwtnn-001 fast-variable-flow
taskset -c 5 bash tools/build-portable.sh axis-index-on core-precision-002 portable-lwtnn-001 fast-variable-flow variable-axis-index
taskset -c 5 bash "$dir/build-fixture.sh" "$fixtures" "$archived_packed_payload"
```

When given the numerical slot, source the release environment and run both axis fixtures with the saved geometry on CPU 3. The baseline and candidate each compare every lookup to an explicit global search. Then run the pinned ROOT/ATLAS histogram oracle for both builds. Both commands must exit successfully and corresponding logs must match.

```bash
source tools/reference/setup.sh
for variant in off on; do
  taskset -c 3 "$fixtures/axis-$variant" < "$fixtures/geometry.txt" > "$fixtures/axis-$variant.log"
  taskset -c 3 "$fixtures/oracle-$variant" > "$fixtures/oracle-$variant.log"
done
cmp "$fixtures/axis-off.log" "$fixtures/axis-on.log"
cmp "$fixtures/oracle-off.log" "$fixtures/oracle-on.log"
```

Use the same precision interpreter and unchanged complete packed payload for the original-file comparison. Validate both providers with the full-interval object comparator and exact small, large and combined histograms. Preserve the same accepted NNJVT option and matched reads in each arm. Compare whole-process fresh starts and full-systematics evaluation with the existing paired controller, keeping both provider choices identical except the new index. The patch and fixture were staged from archived evidence, before any trial build or numerical run.
