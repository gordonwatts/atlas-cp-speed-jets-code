# Histogram status masks

The direct Python evaluator validates the native uncertainty and large-R smearing status codes while preparing every state. The histogram consumes calibrated values, weights, and selection decisions. The uint32 status decorations are used by the diagnostic observer.

This trial leaves the presence conversion and error check in their original order. It gates only the two where operations and uint32 conversion used for the diagnostic decorations. It also allocates the absent large-R smearing decoration only in diagnostic mode. The stock diagnostic arrays and ordering remain exact.

Run from the WSL campaign checkout with the reserved CPU3 slot.

~~~bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 3 .reference-venv/bin/python \
  provenance/provider-patches/visible-status-mask/trial.py \
  results/build/final-native-prefix-001/libjets.so \
  results/diagnostics/visible-status-mask-001
~~~

Add --oracle-only to stop after the synthetic status/error checks and 32 original events per scope in each arm. The bounded comparisons use the clean full-file stock observer.

The runner archives the committed sources and applies the patch only to its candidate snapshot. Both arms use the same provider and packed payloads. Each small-R or combined process uses 1,000-event chunks and signed-state reuse. Each large-R process uses 4,000-event chunks. Four fresh process pairs rotate scope and arm order. The first pair warms each scope and the three retained pairs supply the wall spread checks. Each process must use one worker, map the original file and expected native libraries, emit the archived physical branch digest, and reproduce the full stock histogram. Host and resource receipts accompany every timing.

Single-scope jobs read the corresponding prebuilt packed document. A future installed comparison would include the wheel's startup and temporary extraction.
