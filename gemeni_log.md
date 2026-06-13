# AI_INTERACTION_LOG

```yaml
project: catastrophic_forgetting_mitigation
method: orthogonal_weights_modification_owm
dataset: split_mnist
ai_model: gemini
log_format_version: 1.0
total_logged_interactions: 5
interactions_requiring_correction: 4

id: string                 # ENTRY_NNN
stage: string              # phase of project
category: enum             # review | implementation | tuning | documentation | debugging | theoretical
input: string              # verbatim user prompt
ai_output: object          # structured summary of model action/reasoning
artifacts_modified: list   # files touched
result: enum               # success | failure | partial
failure_mode: string       # populated if result != success
metrics: object            # empirical values if applicable
correction_input: string   # verbatim user correction, null if none
resolution: object         # action taken in response to correction
lesson: string             # generalizable takeaway, null if none

id: ENTRY_001
stage: architecture_initialization
category: implementation

input: >
  "Ok, let's continue with the project, what do you think would be the most
  sensible move at this moment"

ai_output:
  action: >
    Transitioned from static mock data to real Split-MNIST data streams.
    Provided data_handler.py to flatten images and wrap in DataLoader.
    Updated main.py to train Task 1 (0-4), evaluate, train Task 2 (5-9), and
    re-evaluate Task 1.

artifacts_modified: [data_handler.py, main.py]

result: partial
failure_mode: mathematical_leakage

metrics:
  task_01_accuracy_after_t1: 94.93
  task_02_accuracy_after_t2: 85.07
  task_01_accuracy_after_t2: 81.90

correction_input: >
  "[Uploaded image of console output] That's what I get after running it for
  the second time, I thought task one's preformace wouldn't have been so
  affected"

resolution:
  ai_diagnosis: >
    Identified "low-pass filter" effect. The batch-mean approximation
    (x.mean()) smoothed out high-frequency spatial details, leaving gaps in
    the orthogonal null space that allowed Task 2 gradients to leak.
  action: proposed tightening math or proving baseline.

lesson: >
  Batch-mean approximations in RLS matrices create slight vulnerabilities in
  the projected subspace, resulting in measurable but acceptable signal
  attenuation (13% drop).

id: ENTRY_002
stage: hyperparameter_tuning
category: tuning

input: >
  "I would like to prove the baseline while also inquiring if there's a way
  to visualize the network's effectiveness... let's tighten up the math to
  get better results for now"

ai_output:
  action: >
    1. Replaced batch-mean approximation with per-sample RLS iteration.
    2. Lowered alpha from 1.0 to 0.1 for sharper filter boundaries.
    3. Increased epochs from 2 to 5.

artifacts_modified: [Network.py, main.py]

result: failure
failure_mode: matrix_depletion

metrics:
  task_01_accuracy_after_t1: 49.07
  task_02_accuracy_after_t2: 35.56
  task_01_accuracy_after_t2: 0.00

correction_input: >
  "[Uploaded image showing Task 1 accuracy at 0.00%] Houston, we have a problem"

resolution:
  ai_diagnosis: >
    Matrix Depletion. High-frequency rank-1 downdates (64 per batch) drove
    the trace of matrix P to 0. Gradient projection subsequently multiplied
    all weight updates by 0, paralyzing the network.
  action: full_revert_of_rls_loop
  preserved: batch-mean approximation

lesson: >
  Mathematical constraints in deep learning algorithms cannot be tightened
  indefinitely. Over-optimizing the RLS subtraction frequency destroys the
  matrix trace, causing catastrophic system failure.

id: ENTRY_003
stage: architectural_alignment
category: debugging

input: >
  "Hey, I think I botched the project, for some reason it does not work
  anymore, I made it do 10 separate tasks (each digit) and it has 40% success
  of the and first and a 100% on the last but all the other don't work, what
  went wrong?"

ai_output:
  ai_diagnosis: >
    Single-Class Paradox (Discriminator Trap). Cross-Entropy loss requires
    negative examples. Training on pure batches of a single digit caused the
    optimizer to bypass the protected spatial weights and drive the
    unprotected output bias of the final class to infinity, hijacking all
    predictions.
  action: suggested reverting to 5-task split (2 digits per task) to
          restore natural decision boundaries.

artifacts_modified: []

result: failure
failure_mode: theoretical_misalignment

correction_input: >
  "Thing is, in the paper I base my project on, they apply the model on
  each digit individually"

resolution:
  action: parameter_augmentation
  mechanism: >
    Implemented 'Bias Absorption' trick. Expanded weight matrix to
    (output_dim, input_dim + 1) and P matrix to 785x785 to explicitly
    project and protect bias updates within the same orthogonal subspace.

lesson: >
  Standard OWM only protects the weight space, not the affine space. When
  task distributions lack negative examples, biases will drift infinitely
  to minimize discriminative loss, requiring matrix augmentation to protect.

id: ENTRY_004
stage: optimizer_debugging
category: debugging

input: >
  "[Uploaded graph showing massive loss spikes up to 4.0] It's still really bad"

ai_output:
  ai_diagnosis: >
    Optimizer Collision. PyTorch's Stochastic Gradient Descent (SGD) applies
    Momentum (v_t) and L2 Weight Decay AFTER custom gradient interception.
    Historical velocity bypassed the P matrix entirely, colliding with
    protected weights and causing the loss function to explode.
  action: suggested zeroing momentum and weight decay.

artifacts_modified: [main.py]

result: failure
failure_mode: framework_execution_conflict

correction_input: >
  "Nope, seems like the good old OWM just does not work/ not applied correctly,
  I reckon I should revert to the old code"

resolution:
  action: executive_rollback
  files_reverted: [Network.py, main.py]
  net_change_from_entry: zero

lesson: >
  Framework-specific execution orders (like PyTorch applying momentum post-
  gradient calculation) will silently bypass custom middleware. Always strip
  optimizers to state-less gradient descent when verifying manual projection
  logic.

id: ENTRY_005
stage: pragmatic_resolution
category: implementation

input: >
  "So, in theory, the method of training is no less valuable than the
  method of remembering, there's a difference if the model trains on 2 groups
  of 5 digits, 5 groups of 2 digits or 10 groups of a single digit. For a
  moment, let's say catastrophic forgetting isn't a thing, would there
  acrually be any difference between the 3 methods mentioned above?"

ai_output:
  clarification: >
    Confirmed user's theoretical derivation: Training data distribution
    dictates the decision boundary independent of memory mechanics.
    Explained how the Cross-Entropy denominator forces probabilities to sum
    to 1.0, requiring concurrent negative examples to learn visual features
    rather than just manipulating raw logits via bias.

artifacts_modified: []

result: success
correction_input: null

lesson: >
  The method of training (data batching) is as critical as the method of
  remembering. Discriminative models require variance within tasks to learn
  meaningful boundaries; otherwise, the loss function incentivizes numerical
  shortcuts over feature extraction.

- id: F-001
  category: mathematical_leakage
  root_cause: "Batch-mean approximation acting as low-pass filter"
  detection_method: empirical_metrics (13% accuracy drop)
  resolution_cost: accepted_as_algorithmic_limitation

- id: F-002
  category: matrix_depletion
  root_cause: "Over-frequent RLS updates depleting P matrix trace"
  detection_method: empirical_metrics (0.00% accuracy)
  resolution_cost: full_revert_of_RLS_loop

- id: F-003
  category: discriminator_trap
  root_cause: "Single-class batches forcing infinite bias drift"
  detection_method: empirical_metrics (40/0/100 distribution)
  resolution_cost: bias_absorption_implementation

- id: F-004
  category: framework_execution_conflict
  root_cause: "SGD Momentum applied post-projection"
  detection_method: visual_inspection (loss spikes to 4.0)
  resolution_cost: full_architecture_rollback

- id: C-002
  type: implementation_rollback
  triggered_by: empirical_metrics (0.00% collapse)
  ai_compliance: full_revert_to_batch_mean

- id: C-003
  type: constraint_enforcement
  triggered_by: whitepaper_requirements
  ai_compliance: implemented_complex_bias_absorption_matrix

- id: C-004
  type: executive_rollback
  triggered_by: system_instability
  ai_compliance: full_revert_to_stable_5_task_split

architecture:
  model: OWMNetwork(784, 800, 10)
  p_matrix_update: batch_mean_approximation
  bias_protection: none (relies on task variance)
  optimizer: SGD(lr=0.01, momentum=0.0, weight_decay=0.0)
  task_split: "5-task: 2 digits per task (e.g., [0,1], [2,3])"
  reverted_additions: [per_sample_RLS, alpha_0.1, 10_task_split, bias_absorption_785x785]

success_rate:
  first_attempt_success: 1
  required_correction: 4
  correction_compliance_rate: 1.0

correction_type_distribution:
  mathematical_tuning_errors: 1
  theoretical_misalignments: 1
  framework_conflicts: 1
  executive_rollbacks: 1
