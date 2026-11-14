# Superseded first job

Job d39f37aa-6f9 completed, but candidate installation reused a module path
while preserving source timestamps. Python timestamp bytecode caching could
then execute a preceding same-sized candidate. The nominal seeds 1 and 2 and
48-step run demonstrably recorded seed=0 and steps=12 in their artifacts.
These files are retained as diagnostic history, not valid named experiments.
The replacement harness clears only the candidate module bytecode before
installing each candidate, records its source digest, and reruns all cases.
