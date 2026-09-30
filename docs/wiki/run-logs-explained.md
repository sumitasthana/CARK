# Run logs explained

Updated 2026-09-29. This is a record of runs, including failed runs.

**What have we found?** No recorded checkpoint run meets both targets. The
latest 30-step run got task 3 down to 12.6% at step 27. We need 12% or lower.
Task 0 changed by 3.6 percentage points at that step, within its limit.

- [Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index): all 21 records.
- [Rules for reading results](https://github.com/sumitasthana/CARK/wiki/Protocol): what counts as a pass.
- [Latest 30-step run](https://github.com/sumitasthana/CARK/wiki/Experiment-forgetting-30step-20260929): the full results.
- [Source files](https://github.com/sumitasthana/CARK/wiki/Evidence-and-files): where the numbers came from.
- [Software checks](https://github.com/sumitasthana/CARK/wiki/Software-validation): checks of the code.
- [Add a run](https://github.com/sumitasthana/CARK/wiki/Record-template): how to keep the next record.

The original files on Drive are not copied into this wiki. Some numbers came
from tables pasted into the conversation. We keep those numbers as reported and
leave missing values blank. The [registry.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/registry.json) is the
structured record. The [Model diagnostics](https://github.com/sumitasthana/CARK/wiki/Model-diagnostics) page
explains measurements inside the model.
