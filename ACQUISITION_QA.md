# CAWM-P station acquisition QA

## Root cause

The vendored HydroBR implementation converted network timeouts, HTTP/connection failures, malformed XML and legitimate empty responses into the same empty DataFrame. The UI therefore could not explain missing stations. Calls were sequential and could each wait up to 120 seconds. No list truncation, DataFrame display limit, `head()` or IDW rule caused the loss.

The deterministic largest case is ANA outlet `17050001` (ÓBIDOS), 4,670,000 km². With the default 10 km buffer, 915 rain gauges are selected (876 inside and 39 in the buffer); the unchanged IDW weights sum to 1.0.

## Release behavior

- one bounded request attempt at a time in the provider, orchestrated with at most three attempts;
- 30 s explicit request timeout;
- exponential backoff with jitter;
- conservative worker pool (default 8);
- one terminal receipt row per selected request, including duplicates;
- `no_data` is distinct from `failed_after_retries`;
- partial completion is labelled **Completed with warnings** and remains downloadable.

The live service was not required for acceptance. Mocks cover 1, 24 and 400 stations, intermittent success, timeout/failure, legitimate no-data, duplicates and partial batches. Run `submission_release/supplementary/run_largest_basin_acquisition_diagnostic.py` locally to consult ANA and produce a real receipt when network policy and service availability permit.

