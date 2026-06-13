# AI_INTERACTION_LOG

```yaml
project: catastrophic_forgetting_mitigation
method: orthogonal_weights_modification_owm
paper_reference: zeng_et_al_2019_arxiv_1810.01256
dataset: split_mnist_2task
ai_model: claude_sonnet_4
log_format_version: 1.0
total_logged_interactions: 9
interactions_requiring_correction: 5
```

## SCHEMA

Each entry uses the following fields:

```yaml
id: string                 # ENTRY_NNN
stage: string               # phase of project
category: enum               # review | implementation | tuning | documentation | debugging | theoretical
input: string                # verbatim user prompt
ai_output: object            # structured summary of model action/reasoning
artifacts_modified: list      # files touched
result: enum                  # success | failure | partial
failure_mode: string          # populated if result != success
metrics: object               # empirical values if applicable
correction_input: string      # verbatim user correction, null if none
resolution: object            # action taken in response to correction
lesson: string                # generalizable takeaway, null if none
```

---

## ENTRY_001

```yaml
id: ENTRY_001
stage: initial_review
category: review

input: >
  "Hey mate, here's another project I have, this time we are using deep
  learning and neural networks. Attached are the instructions and the code
  I already have. We are supposed to 'solve' Catastrophic Forgetting. I
  wanted to find a unique and niche way to solve this, so I went the OWM way.
  Tell me what you think, what could use a touch up, what's already fine and
  everything you think is worth mentioning, as before, you are permitted to
  change anything you want."

ai_output:
  findings:
    eval_on_train_split: true        # severity: high, category: methodology
    hardcoded_device: "cuda"         # severity: medium, category: portability
    missing_baseline_model: true     # severity: high, category: experimental_design
    missing_visualization: true      # severity: medium, category: deliverable_requirement
  verified_correct:
    - rls_projector_update_formula
    - gradient_projection_formula

artifacts_modified: [data_handler.py, Network.py, main.py]

changes:
  - file: data_handler.py
    diff: "added train: bool = True param, passed to torchvision.datasets.MNIST(train=...)"
  - file: Network.py
    diff: "added BaselineNetwork class (plain nn.Linear x2, no P matrix, no projection)"
  - file: main.py
    diff: "torch.device(cuda_if_available_else_cpu); added plot_results(); added baseline training run"

result: success
correction_input: null
lesson: null
```

---

## ENTRY_002

```yaml
id: ENTRY_002
stage: paper_consultation
category: implementation

input: >
  "Great, by the way. Here's the paper I found that explains the method
  https://arxiv.org/pdf/1810.01256. Think there's anything noteworthy in
  there that I should replicate in my code?"

ai_output:
  source_fetched: "arxiv.org/pdf/1810.01256"
  discrepancies_identified:
    optimizer:
      paper: sgd_with_momentum_0.9
      code: sgd_plain
    regularization:
      paper: l2_0.001_plus_dropout_0.2
      code: none
    task_structure:
      paper: 10_single_digit_tasks
      code: 2_task_split_[0-4]_[5-9]
    repetition:
      paper: mean_std_over_10_runs
      code: single_run
    alpha_schedule:
      paper: "decaying, alpha0 * lambda^(i/n_batches)"
      code: fixed_0.75
  action: implemented_all_discrepancies_simultaneously

artifacts_modified: [Network.py, main.py, data_handler.py]

changes:
  - file: Network.py
    diff: "added nn.Dropout(0.2); added alpha_override param for decay schedule"
  - file: main.py
    diff: "TASKS=[[d] for d in range(10)]; N_REPEATS=10; SGD(momentum=0.9, weight_decay=0.001)"

result: failure
failure_mode: architectural_incompatibility

failure_details:
  root_cause: "output_layer_dim=10 (one neuron per digit class 0-9)"
  mechanism: >
    single-digit tasks provide gradient signal to exactly 1 of 10 output
    neurons per task. 9/10 output neurons receive zero gradient signal per
    task. sequential single-class training leaves only the most-recently-
    trained neuron with useful weights.

metrics:
  task_01_accuracy: 40.0
  task_02_through_09_accuracy: ~0.0
  task_10_accuracy: 100.0

correction_input: >
  "Hey, seems like these changes only messed up the code, now I has a 40%
  success rate on the first task, a 100% on the last and the others at 0%.
  I think it's best we stick with what I already wrote on my own but with a
  bit of modification, the code you suggested also seemed way longer and
  more complex which is not what I am trying to achieve"

resolution:
  action: full_revert
  files_reverted: [Network.py, main.py, data_handler.py]
  net_change_from_entry: zero
  preserved: "2-task split [0,1,2,3,4] vs [5,6,7,8,9]"

lesson: >
  paper_fidelity != implementation_correctness. paper experimental configs
  encode architecture-specific assumptions (output_dim, task_granularity)
  not transferable without modification. AI should validate paper-derived
  changes against existing architecture constraints BEFORE implementation.
```

---

## ENTRY_003

```yaml
id: ENTRY_003
stage: hyperparameter_tuning
category: tuning

input: >
  "Is there a semi easy way to up the OWM performance of task 1 after task 2
  without too much harming task 2 performance?"

ai_output:
  suggestions_ranked:
    - rank: 1
      action: lower_alpha
      from: 0.75
      to: [0.1, 0.01]
      mechanism: "smaller denominator -> larger P correction per batch -> faster subspace closure"
    - rank: 2
      action: increase_task1_epochs
      mechanism: "more projector updates fire before task2 begins"
    - rank: 3
      action: experience_replay
      mechanism: "mix task1 batches into task2 training loop"
  recommended_combination: [1, 2]

result: partial

metrics_reported_by_user:
  task1_retention: improved
  task2_accuracy: degraded

correction_input: >
  "This really did dampen the accuracy of task 2, I wonder, would having a
  bigger network solve this issue?"

resolution:
  proposed_by: user
  action: increase_hidden_dim
  from: 800
  to: [2000, 4000]
  ai_confirmation:
    mechanism: >
      P shape = (hidden_dim, hidden_dim). null_space(P) scales with
      hidden_dim. task1 claims a proportional subspace but absolute
      remaining dimensions increase with hidden_dim, giving task2 more
      orthogonal directions for gradient updates without weakening task1
      protection.
  implemented_in_session: false  # suggestion only

lesson: >
  alpha and epoch count operate on a FIXED null_space and therefore have a
  hard tradeoff ceiling (zero-sum between task1 protection and task2
  plasticity). hidden_dim is the only lever that increases TOTAL available
  null_space rather than redistributing a fixed quantity.
```

---

## ENTRY_004

```yaml
id: ENTRY_004
stage: documentation
category: scope_interpretation

input: >
  "Great, I think it would be better if you could add a comment that
  explains every line in the code so I could submit it to the repo"

ai_output:
  target_resolved_as: code_snippets_inside_algorithmic_thinking.md
  action: added line-by-line comments to 3 illustrative snippets within the markdown document

result: failure
failure_mode: scope_misinterpretation

failure_details:
  user_intent: actual_source_files [main.py, Network.py, data_handler.py]
  ai_target: markdown_documentation_excerpts [subset, illustrative only]
  ambiguity_source: "'the code' has two valid referents when both source files and docs contain code blocks"

correction_input: >
  "Not what I meant, I meant the actual code not the one in README file"

resolution:
  action: rewrite_source_files
  files: [main.py, Network.py, data_handler.py]
  functional_change: none
  comment_coverage: per_line

lesson: >
  disambiguate file-path target before bulk edits when multiple candidate
  artifacts (source vs documentation) contain overlapping content.
```

---

## ENTRY_005

```yaml
id: ENTRY_005
stage: debugging
category: debugging

input: >
  [image: terminal traceback]
  "TypeError: only 0-dimensional arrays can be converted to Python scalars"
  File demo.py, line 25, in update_projector
      xPx = float(x_mean.T @ Px)

ai_output:
  root_cause: "numpy-version-dependent behavior of float() on (1,1)-shaped ndarray"
  fix: "float(x_mean.T @ Px) -> (x_mean.T @ Px).item()"

artifacts_modified: [projector_demo.py]

diff:
  before: "xPx = float(x_mean.T @ Px)"
  after:  "xPx = (x_mean.T @ Px).item()"

result: success
correction_input: "[traceback image]"

lesson: >
  .item() is the version-agnostic scalar extraction method for numpy/torch
  (1,1) arrays; float() is not guaranteed to accept non-0-d arrays across
  numpy versions.
```

---

## ENTRY_006

```yaml
id: ENTRY_006
stage: debugging
category: debugging

input: >
  [image: rendered matplotlib figure]
  "This is what I get, seems like things are lapping each other and the text
  at the top is cut off"

ai_output:
  defects_identified:
    - id: defect_1
      description: "4th heatmap subplot overlapped by colorbar"
      cause: "fig.colorbar(im, ax=fig.axes[:4]) shares space with last heatmap axis"
    - id: defect_2
      description: "suptitle clipped at top of figure"
      cause: "fig.suptitle(..., y=1.01) exceeds savefig bbox"

artifacts_modified: [projector_demo.py]

diff:
  - change: "GridSpec(1,4) -> GridSpec(2,5)"
    reason: "reserve dedicated column[4] for colorbar via hidden axes"
  - change: "fig.suptitle(..., y=1.01) -> fig.suptitle(...)"
    reason: "remove out-of-bounds y offset"

result: success
correction_input: "[image showing subplot overlap and clipped title]"

lesson: >
  matplotlib colorbar placement via ax=<shared_axes_list> can collide with
  adjacent subplots; isolate colorbar to a dedicated reserved axis.
  suptitle y > 1.0 can render outside savefig bbox; omit unless validated
  against figsize.
```

---

## ENTRY_007

```yaml
id: ENTRY_007
stage: theoretical_discussion
category: theoretical

input: >
  "I don't really get the representative input vector, what do you mean when
  you say we 'we average them into one representative vector'?, if we
  average all the images are how are we supposed to distinguish between one
  over another?"

ai_output:
  clarification:
    p_update_purpose: subspace_membership_recording   # NOT classification
    batch_mean_role: approximates_cluster_centroid_of_task_input_distribution
    individual_sample_identity: not_required_for_subspace_blocking

result: success
correction_input: null
lesson: null
```

---

## ENTRY_008

```yaml
id: ENTRY_008
stage: theoretical_discussion
category: theoretical

input: >
  "Ok, so how does task 1 differentiate from task 2 in that regard?, task 2
  is also quite similar to task 1 and numbers that were chosen could have
  been shuffled. What I am trying to say is that by task 1 and 2 being so
  similar, most of the directions are either almost locked off while others
  are in little need to change as the weights were already set during task 1"

ai_output:
  concession: "split-mnist tasks share low-level statistical structure (stroke width, centering, intensity distribution)"
  implications:
    - "P may over-block directions task2 needs, independent of task1 relevance"
    - "weights from task1 partially transferable -> task2 requires less gradient movement -> over-blocking less consequential"
  framing_offered: >
    "split-mnist does not stress-test OWM's geometric limits"
    (this framing was challenged in ENTRY_009)

result: success
correction_input: null
note: "framing later revised — see ENTRY_009"
```

---

## ENTRY_009

```yaml
id: ENTRY_009
stage: theoretical_discussion
category: theoretical_overreach

input: >
  "I don't fully agree we your statement, while OWM could have had it's
  limits tested a bit further, it is factually better than having nothing at
  all. Even the tasks are similar, task 1's accuracy after task 2 for the
  regular network was nearing the zero percent so OWM does do some heavy
  lifting"

ai_output_prior_entry_008:
  implied: "task similarity diminishes OWM's measured contribution"

result: failure
failure_mode: theoretical_overreach

failure_details:
  empirical_baseline_task1_after_task2: "~0%"
  empirical_owm_task1_after_task2: "significantly_above_baseline"
  gap_attribution: "100% attributable to OWM mechanism, independent of task similarity"
  conflation_error: >
    entry_008 conflated two independent claims:
    (a) "tasks are statistically similar" [true]
    (b) "OWM's measured effect size on this benchmark is small" [false, contradicted by data]

correction_input: >
  "I don't fully agree we your statement, while OWM could have had it's
  limits tested a bit further, it is factually better than having nothing at
  all. Even the tasks are similar, task 1's accuracy after task 2 for the
  regular network was nearing the zero percent so OWM does do some heavy
  lifting"

resolution:
  revised_framing: >
    "split-mnist doesn't fully stress-test OWM's generalizability to
    dissimilar-task settings" [claim about external validity]
  retracted_framing: >
    "OWM's measured effect on split-mnist is small" [claim about this
    experiment's results]

lesson: >
  a critique of generalizability (does this benchmark probe the hard case?)
  is logically independent from a critique of effect size (did the method
  work in THIS experiment?). AI conflated the two; correction required
  domain-specific empirical reasoning, not code inspection. this is a
  correction of REASONING, not implementation.
```

---

## FAILURE_REGISTRY

```yaml
- id: F-002
  category: architectural_incompatibility
  root_cause: "output_dim=10 incompatible with single-class task partitioning"
  detection_method: empirical_metrics  # 40/0/100 accuracy distribution
  resolution_cost: full_revert_3_files

- id: F-004
  category: scope_misinterpretation
  root_cause: "ambiguous referent ('the code') resolved to wrong artifact type"
  detection_method: user_inspection
  resolution_cost: rewrite_3_files_comments_only

- id: F-005
  category: runtime_error
  root_cause: "numpy float() incompatible with (1,1)-shaped array"
  detection_method: traceback
  resolution_cost: one_line_fix

- id: F-006
  category: layout_defect
  root_cause: "matplotlib colorbar/axes collision + out-of-bounds suptitle offset"
  detection_method: visual_inspection
  resolution_cost: gridspec_restructure

- id: F-009
  category: theoretical_overreach
  root_cause: "conflated generalizability critique with effect-size critique"
  detection_method: user_domain_reasoning
  resolution_cost: reframing_no_code_change
```

## CORRECTION_REGISTRY

```yaml
- id: C-002
  type: implementation_rollback
  triggered_by: empirical_metrics
  ai_compliance: full

- id: C-003
  type: user_initiated_alternative
  triggered_by: empirical_metrics
  ai_compliance: adopted_and_extended_with_mechanism_explanation

- id: C-004
  type: scope_redirect
  triggered_by: explicit_statement
  ai_compliance: full

- id: C-005
  type: bug_report_with_traceback
  triggered_by: runtime_error
  ai_compliance: full

- id: C-006
  type: bug_report_with_screenshot
  triggered_by: visual_defect
  ai_compliance: full

- id: C-009
  type: reasoning_correction
  triggered_by: domain_knowledge
  ai_compliance: full
```

## FINAL_STATE

```yaml
architecture:
  model: OWMNetwork(784, 800, 10)
  alpha: 0.75  # fixed, no decay
  baseline: BaselineNetwork(784, 800, 10)  # identical dims, no P matrix
  optimizer: SGD(lr=0.01)  # no momentum, no weight_decay
  regularization: none  # no dropout
  task_split: "2-task: digits[0-4] vs digits[5-9]"
  reverted_additions: [10_task_split, momentum, l2, dropout, alpha_decay, n_repeats_10]

success_rate:
  first_attempt_success: 4   # ENTRY_001, ENTRY_007, ENTRY_008(initial)
  required_correction: 5     # ENTRY_002, 004, 005, 006, 009
  correction_compliance_rate: 1.0  # 5/5 corrections fully applied

correction_type_distribution:
  implementation_bugs: 3   # 002, 005, 006
  scope_ambiguity: 1        # 004
  reasoning_error: 1        # 009
```
