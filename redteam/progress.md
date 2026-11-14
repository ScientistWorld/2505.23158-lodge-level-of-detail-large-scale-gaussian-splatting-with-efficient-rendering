# Audit progress

Job 43e9e480-b5b completed the cached reference, three independently rebuilt
12-step seeds, and 48/120-step variants. The 1000-step variant repeatedly
restarted because workers drained after about 26 minutes. It was cancelled
before completion; it has no valid score yet.

The next job preserves each completed deterministic fine-tuning round under
redteam/_cache/, keyed by all function arguments and implementation bytes.
This changes resumption only, not optimization or selection. The full
1000-step variant can now finish across worker drains. Remaining ablations,
exhaustive threshold search, and shortcut probes follow in the same job.

Earlier first-job results in first_job_cache_bug/ are invalid comparisons:
timestamp-based bytecode reused same-length candidate wrappers. The harness
now removes only its own candidate bytecode before installation, records source
hashes and artifact configuration, and checks seed/step identities.

Job 01fee188-6ff completed the 1000-step variant and both weaker variants.
The worker then died and remained inactive; the job was cancelled. The
1000-step chunked renderer loses quality (25.8267 dB versus 27.8194 rebuilt
seed 0), while dynamic LOD reaches 28.1590 dB. This is retained as evidence,
not discarded. The next job skips these completed cases and resumes the
remaining short checks. Exhaustive search now saves completed pairs and
prints progress. Failed-test error tails are retained with score provenance.
