# Reproduction Checklist (iteration 11)

- [x] Paper read; method restored to the paper's steps (Eq. 2–5, supp. A.2)
- [x] Hidden world generator and test targets only in `eval/test/`
- [x] Public data committed in `data/lodge_city/`
- [x] Evaluator renders representations itself; costs recomputed; nothing self-reported is read
- [x] Fixed baseline, gates, worst values on failure, every method on disk scored
- [x] Reference ×3 seeds, better and weaker variants, rules and shortcuts run through the hidden test
- [x] `scoring/reference.json`, `scores_{train,test}.json`, `EXPERIMENTS.md`, `TARGETS.md`, `CONSTRAINTS.md`, `DIRECTION.md`
- [x] `briefing/` updated
- [x] `python validate.py --compare --post-design` passes
- [x] HARDENING.md written
