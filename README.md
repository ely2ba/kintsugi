# Kintsugi v2

Does repeated behavioral repair preserve a language model's ability to keep learning?

This is the corrected experiment following the closed
[Kintsugi v1 study](https://github.com/ely2ba/kintsugi-v1/tree/v1-final).
The [scientific contract](SPEC.md) fixes seven heterogeneous tasks, four
counterbalanced orders, and twelve lineages. Learning must produce both new-task
competence and measured instruction-following damage; repair is then evaluated
for visible recovery, future learning, retention, diversity, and lifecycle cost.

Status: repository transition complete; M0/M1 runner implemented and tested locally.
The acquisition reference sweep now uses the registered 120-update budget, with
independent gate and held-out reference scores. No paid v2 calls or M2 execution
have started. M1 must pass the registered task,
persistence, probe-coverage, and diversity gates. A successful M1 ends with a
launch packet and a separate human authorization before M2.

The local implementation includes task generators and exact checkers, fixed order
manifests, masked training and repair, checkpoint recovery, and tests for stopping,
probe clocks, retention and coverage. The measurement-noise rule is now frozen;
M1 execution is restricted to its public tested-input freeze and a separate v2 project.

See [OPERATIONS.md](OPERATIONS.md) for local checks and execution boundaries.
See [PROVENANCE.md](PROVENANCE.md) for the exact archived predecessor.
