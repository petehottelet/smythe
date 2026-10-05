# Support matrix

Smythe has three execution paths:

- **Ordinary runs**: `Swarm` without a `run_store`.
- **Durable text workflows**: `Swarm(run_store=SQLiteWorkflowStore(...))`.
  See [durable text workflows](workflow-accounting.md).
- **Jobs**: manifest-declared artifact operations. See [Jobs](jobs.md).

This page lists what each path accepts. Each supported cell names a test
that exercises it, and each refusal names a test that shows when it happens.
`tests/test_support_matrix_docs.py` checks that every cited test exists.

## Providers on ordinary runs

| Provider | Text | Tools | Image inputs | Image output |
|---|---|---|---|---|
| `OfflineProvider` | Yes | Rejected before planning | Acknowledged in the text only | 1x1 PNGs with `artifacts_per_call` |
| `AnthropicProvider` | Yes | Yes | Yes | No |
| `OpenAIProvider` | Yes | Yes, non-strict | Yes, as data URLs | No |
| `OpenAIImageProvider` | Prompts to `gpt-image-*` models only | Rejected before planning | Rejected before node calls | Yes |
| `GeminiProvider` | Yes | Yes | Yes | Image models, or explicit `response_modalities` |
| `OpenAIResponsesProvider` | `gpt-6-astra`, `gpt-5.6-sol` | Yes, non-strict | Rejected before node calls | No |
| `AnthropicMessagesProvider` | `claude-fable-5-1` | Rejected before planning | Rejected before node calls | No |
| Custom `Provider` | Yes, through `complete()` | Only if it overrides `chat()` | Only if it overrides `chat()` | If it returns `artifacts` |

No provider supports structured output or streaming. Planners ask for JSON
and parse it from the reply text. Test:
`tests/test_support_matrix.py::test_no_provider_offers_structured_output_or_streaming`.

Each built-in provider declares these features for a model through
`Provider.feature_support(model)`, which returns a `ProviderFeatures`. An
ordinary run reads the declaration before any provider call and raises
`ValueError` for a feature declared unsupported:

- A `tool_runtime` with a provider that cannot call tools is rejected
  before routing or planning, even when no node would be offered a tool.
- A model the provider does not accept, or a node with
  `attach_dep_artifacts` on a provider that cannot take image inputs, is
  rejected before the first node call. A caller-built graph is checked
  before it runs and a generated plan as soon as it exists, since only the
  plan says whether it wants image inputs. A generated plan's nodes all use
  the run's model, so with `LLMArchitect` planning that model is checked
  before the planning call.
- A resumed run checks its unfinished nodes before calling any of them.

A custom `Provider` declares nothing unless it overrides
`feature_support()`, and neither does a subclass that replaces a built-in
provider's `chat()`, so ordinary runs refuse nothing about them in advance.
Their calls fail as before: a provider without tool support raises
`NotImplementedError` for a node with tools.

Tests:

- Declared features: `tests/test_support_matrix.py::test_providers_declare_what_each_model_supports`.
- Refused before any provider call, planning included:
  `tests/test_support_matrix.py::test_ordinary_run_rejects_a_tool_runtime_its_provider_cannot_serve_before_any_call`,
  `tests/test_support_matrix.py::test_tool_runtime_is_checked_before_an_overridden_plan_runs`,
  `tests/test_support_matrix.py::test_ordinary_run_rejects_image_inputs_its_provider_cannot_take_before_any_call`,
  `tests/test_support_matrix.py::test_ordinary_run_rejects_a_model_its_provider_does_not_support_before_any_call`
  and `tests/test_support_matrix.py::test_resume_checks_pending_nodes_before_any_call`.
- Refused after planning, before any node call, and resumed without
  planning again: `tests/test_support_matrix.py::test_generated_plan_with_image_inputs_is_rejected_before_any_node_call`
  and `tests/test_support_matrix.py::test_a_plan_refused_before_its_nodes_resumes_without_planning_again`.
- Providers that declare nothing still run:
  `tests/test_support_matrix.py::test_providers_that_declare_nothing_still_run_with_tools_image_inputs_and_a_budget`.
- `OfflineProvider`: text `tests/test_offline_provider.py::test_full_swarm_pipeline_runs_offline`;
  image inputs `tests/test_provider.py::test_offline_provider_acknowledges_attachments`;
  image output `tests/test_provider.py::test_offline_provider_artifacts_per_call`.
- `AnthropicProvider`: text `tests/test_provider.py::test_anthropic_guard_non_text_content`;
  tools `tests/test_provider.py::test_anthropic_chat_parses_tool_use_blocks` and
  `tests/test_provider.py::test_anthropic_tool_results_go_in_one_user_message`;
  image inputs `tests/test_provider.py::test_anthropic_maps_attachments_to_image_blocks`.
- `OpenAIProvider`: text `tests/test_provider.py::test_openai_without_a_refusal_keeps_text_and_tool_call_stop_reasons`;
  tools `tests/test_provider.py::test_openai_chat_parses_tool_calls` and
  `tests/test_provider.py::test_openai_tool_results_become_tool_role_messages`;
  image inputs `tests/test_provider.py::test_openai_maps_attachments_to_image_url_parts`.
- `OpenAIImageProvider`: image output `tests/test_provider.py::test_openai_image_provider_extracts_artifacts_usage_and_cost`;
  tools rejected `tests/test_provider.py::test_openai_image_provider_rejects_non_image_model_and_tools`;
  image inputs rejected `tests/test_provider.py::test_openai_image_rejects_attachments_instead_of_dropping_them`.
- `GeminiProvider`: text `tests/test_provider.py::test_gemini_provider_complete`;
  tools `tests/test_provider.py::test_gemini_chat_synthesizes_call_ids` and
  `tests/test_provider.py::test_gemini_tool_results_matched_by_name`;
  image inputs `tests/test_provider.py::test_gemini_maps_attachments_to_inline_data_parts`;
  image output `tests/test_provider.py::test_gemini_extracts_image_artifacts` and
  `tests/test_provider.py::test_gemini_auto_requests_image_modalities_for_image_models`.
- `OpenAIResponsesProvider`: text `tests/test_provider_responses.py::test_complete_one_generation_no_automatic_quote`;
  tools `tests/test_provider_responses.py::test_tool_history_replays_all_items_once_and_uses_call_id`;
  image inputs rejected `tests/test_provider_responses.py::test_unsupported_inputs_reject_before_http`.
- `AnthropicMessagesProvider`: text in a full ordinary run
  `tests/test_support_matrix.py::test_anthropic_messages_provider_runs_an_ordinary_swarm`;
  tools and image inputs rejected
  `tests/test_support_matrix.py::test_anthropic_messages_provider_rejects_tools_and_attachments_before_any_request`.
- Custom `Provider`: text `tests/test_provider.py::test_base_chat_delegates_simple_case_to_complete`;
  image output `tests/test_executor_base.py::test_executor_persists_artifacts_end_to_end`;
  tools without `chat()` `tests/test_provider.py::test_base_chat_with_tools_raises_not_implemented`;
  image inputs without `chat()` `tests/test_provider.py::test_base_chat_rejects_attachments_for_complete_only_providers`.

## Paths, accounting and retries by provider

| Provider | Durable text workflows | Jobs profile | Cost recorded | Retries inside the SDK |
|---|---|---|---|---|
| `OfflineProvider` | Static plan and echo only | `offline` | Ordinary: blended token estimate, flagged by `Sentinel.cost_contains_token_estimates`. Durable and Jobs: zero | None |
| `AnthropicProvider` | Rejected | None | Blended token estimate, flagged by `Sentinel.cost_contains_token_estimates`; `cost_contains_estimates` stays `False` | The Anthropic SDK default |
| `OpenAIProvider` | Rejected | None | Blended token estimate, flagged by `Sentinel.cost_contains_token_estimates`; `cost_contains_estimates` stays `False` | The OpenAI SDK default, unless `max_retries` is set |
| `OpenAIImageProvider` | Rejected | `openai_image` | Per-call ceiling; without one, a per-image estimate marked incomplete, or unknown | The OpenAI SDK default, unless `max_retries` is set. Jobs sets 0 |
| `GeminiProvider` | Rejected | `gemini_image` | Text: blended token estimate, flagged by `Sentinel.cost_contains_token_estimates`; `cost_contains_estimates` stays `False`. Images: as for `OpenAIImageProvider` | None configured |
| `OpenAIResponsesProvider` | `gpt-6-astra`, `gpt-5.6-sol` | None | Exact native price | None; a client with retries is refused |
| `AnthropicMessagesProvider` | `claude-fable-5-1` | None | Exact native price | None; a client with retries is refused |
| Custom `Provider` | Rejected | None | Its own `cost_usd`, or a blended token estimate flagged by `Sentinel.cost_contains_token_estimates` | Its own |

Each path keeps money in its own unit:

- **Ordinary runs** record float USD in `Sentinel`. The ledger covers execution
  and synthesis (`cost_scope="execution_and_synthesis"`); routing, planning
  and successful supervision calls are outside it. With `max_budget_usd`
  set, `OpenAIImageProvider`, `OpenAIResponsesProvider`,
  `AnthropicMessagesProvider` and image-model `GeminiProvider` calls need
  `max_cost_per_call_usd`, or a node's own `estimated_cost_usd`. Without
  either, the run raises `BudgetEstimateRequired` before any provider call:
  before routing or planning when `execute(Task)` has `LLMArchitect` plan
  the run or an `LLM_MERGE` synthesis needs the ceiling, and before the
  first node call otherwise. `plan()` returns a generated plan whose nodes
  can then be given estimates. See [cost guardrails](budgets.md).
  The native providers, `OpenAIResponsesProvider` and
  `AnthropicMessagesProvider`, record an unresolved bill as unknown and keep
  its reservation. The other providers' token-priced costs are blended
  estimates: `Sentinel.cost_contains_token_estimates` reports them, and
  `cost_contains_estimates` stays `False`. A reply without usage counts as
  zero tokens, a known charge, unless `AnthropicProvider`, `OpenAIProvider`
  or `GeminiProvider` was built with `missing_usage="unknown"`. Its cost is
  then unknown. A reservation held for the call is kept as its estimated
  charge, so `cost_contains_estimates` becomes `True` and `cost_is_complete`
  stays `True`; only a call without one makes `cost_is_complete` `False`.
  The async executor behind parallel runs, `execute_async()` and resume
  reserves before each node's call, as does an `LLM_MERGE` synthesis and a
  serial node whose own or provider estimate is set; a serial `execute()`
  node without one reserves nothing (see [cost guardrails](budgets.md)). A
  call to one of them that raises or times out releases its reservation and
  records no cost.
- **Durable text workflows** record integer nanoUSD in the journal for every
  phase (`cost_scope="complete_text_workflow"`), from request-bound quotes.
- **Jobs** reserve and settle integer micro-USD against each profile's
  inclusive per-call ceiling. Offline profiles require a zero ceiling.

Who calls again after a failure depends on the path and the provider:

- **Ordinary runs, other providers**: the node failure policy
  (`max_retries`, `retry_backoff_s`) decides, including after a refused or
  truncated reply.
- **Ordinary runs, native providers**: once a request is attempted, no node
  policy calls again. An HTTP error including a 429, a transport failure, a
  refusal, a truncation or an unresolved bill raises `ProviderResponseError`
  (`ProviderAccountingError` when the bill is unresolved) and stops the run.
- **Durable text workflows**: a 401, 403, 404, 413 or 429 response settles at
  zero cost, and the node failure policy decides whether to call again.
  Other HTTP or transport failures leave the bill unknown and block the run;
  a refusal or truncation is terminal.
- **Durable read steps** make one request per journaled call. A failed free
  read settles at zero cost and the node failure policy decides whether to
  read again, unless no response arrived and its profile does not declare it
  safe to send again. That read, and a metered read without a 200 response,
  keep unknown exposure and block the run. See
  [recorded read steps](workflow-accounting.md#recorded-read-steps).
- **Jobs** never retry on their own: an operator reroll makes a new attempt,
  up to `max_attempts`.

The last column of the table lists retries an SDK makes inside one call,
which Smythe neither sees nor records.

Tests:

- Durable providers: `tests/test_workflow_runtime.py::test_plan_execute_resume_preserves_task_run_and_local_cost_scope`
  (`OfflineProvider`),
  `tests/test_workflow_native_runtime.py::test_every_builtin_paid_phase_is_counted_once_and_consumed`
  (`OpenAIResponsesProvider`),
  `tests/test_native_messages.py::test_durable_swarm_retries_rate_limited_node_with_exact_totals`
  (`AnthropicMessagesProvider`).
- Durable rejections, before planning, with no provider call and no run
  created: `tests/test_support_matrix.py::test_durable_run_rejects_provider_before_any_call`,
  with one case each for `AnthropicProvider`, `OpenAIProvider`,
  `OpenAIImageProvider`, `GeminiProvider` text and image models, scripted or
  image-producing `OfflineProvider`, a custom `Provider`, subclasses of the
  three supported providers, and an unsupported model.
- Jobs profiles: `tests/test_jobs_runner.py::test_runner_executes_bounded_offline_job_and_persists_artifacts`,
  `tests/test_jobs_runtime_support.py::test_provider_preflight_accepts_complete_local_openai_setup`,
  `tests/test_jobs_runtime_support.py::test_gemini_image_profile_accepts_raster_artifacts_and_reference_images`.
- Cost: `tests/test_budget.py::test_budget_tracks_cost`,
  `tests/test_budget.py::test_explicit_cost_overrides_token_math`,
  `tests/test_provider.py::test_openai_image_records_ceiling_when_invoice_cost_is_unavailable`,
  `tests/test_provider.py::test_unpriced_openai_image_marks_usd_cost_unknown`,
  `tests/test_provider.py::test_gemini_records_ceiling_and_marks_unpriced_images_unknown`,
  `tests/test_provider_responses.py::test_complete_one_generation_no_automatic_quote`,
  `tests/test_workflow_provider.py::test_offline_zero_cost_static_plan_and_echo_no_network`,
  `tests/test_jobs_models.py::test_money_normalizes_to_exact_microdollars`,
  `tests/test_jobs_preflight.py::test_offline_plan_requires_zero_ceiling_and_budget`.
- Uncertain ordinary-run costs: `tests/test_support_matrix.py::test_ordinary_ledger_does_not_mark_uncertain_costs_from_other_providers`,
  `tests/test_budget.py::test_blended_token_cost_is_reported_as_a_token_estimate`,
  `tests/test_provider.py::test_run_with_missing_usage_reports_incomplete_cost_only_when_opted_in`,
  `tests/test_provider.py::test_parallel_run_keeps_the_reservation_of_a_call_without_usage`,
  `tests/test_provider.py::test_serial_node_estimate_stands_in_for_a_call_without_usage`,
  `tests/test_execution_failure_policies.py::test_terminal_failure_stops_queued_work_and_preserves_original_exception`,
  `tests/test_execution_failure_policies.py::test_timeout_obeys_failure_policy_and_settles_before_return`
  and, for the native providers,
  `tests/test_native_response_accounting.py::test_native_failure_is_terminal_and_preserves_exposure`.
- Price ceilings for capped runs: `tests/test_provider.py::test_openai_image_hard_budget_requires_inclusive_call_ceiling`,
  `tests/test_provider.py::test_gemini_hard_budget_uses_only_inclusive_call_ceiling`,
  `tests/test_provider_responses.py::test_legacy_budget_requires_inclusive_explicit_ceiling`;
  refused before any provider call, planning included,
  `tests/test_support_matrix.py::test_ordinary_run_rejects_a_missing_cost_ceiling_before_any_call`,
  `tests/test_support_matrix.py::test_ordinary_run_checks_a_routed_plan_before_the_classifier_call`
  and `tests/test_support_matrix.py::test_ordinary_run_rejects_an_llm_merge_it_cannot_budget_before_any_call`;
  planned first and run with node estimates,
  `tests/test_support_matrix.py::test_plan_returns_a_graph_whose_node_estimates_stand_in_for_a_missing_ceiling`.
- SDK retries: `tests/test_support_matrix.py::test_legacy_provider_clients_leave_retries_to_the_sdk_configuration`,
  `tests/test_support_matrix.py::test_native_providers_refuse_a_client_with_sdk_retries`,
  `tests/test_jobs_runtime_support.py::test_durable_openai_pool_disables_sdk_retries_and_sets_timeout`.
- Ordinary runs, other providers: `tests/test_execution_failure_policies.py::test_consumed_failure_policy_allows_later_work`,
  `tests/test_execution_failure_policies.py::test_truncated_output_is_a_billed_failure_under_the_node_policy`,
  `tests/test_execution_failure_policies.py::test_refused_output_is_a_billed_failure_under_the_node_policy`.
- Ordinary runs, native providers: `tests/test_support_matrix.py::test_native_provider_failure_on_an_ordinary_run_is_terminal_under_a_retry_policy`,
  `tests/test_native_response_accounting.py::test_native_failure_is_terminal_and_preserves_exposure`.
- Durable text workflows: `tests/test_workflow_native_runtime.py::test_rate_limited_worker_retries_under_retry_policy_with_exact_totals`,
  `tests/test_native_messages.py::test_durable_swarm_retries_rate_limited_node_with_exact_totals`,
  `tests/test_workflow_store.py::test_pre_generation_rejection_settles_at_zero_cost_and_keeps_admission_open`,
  `tests/test_workflow_store.py::test_uncertain_http_outcomes_keep_unknown_exposure`,
  `tests/test_native_messages.py::test_server_errors_including_overloaded_keep_unknown_exposure`,
  `tests/test_native_messages.py::test_truncations_refusals_keep_their_known_charge`,
  `tests/test_workflow_provider.py::test_known_bad_output_is_paid_terminal_and_replayed_locally`.
- Durable read steps: `tests/test_workflow_read_tool.py::test_a_failed_read_follows_its_node_failure_policy`,
  `tests/test_workflow_read_tool.py::test_reads_settle_by_their_declared_charge_and_outcome`.
- Jobs: `tests/test_jobs_runner.py::test_selective_reroll_replaces_only_rejected_operation`.

## Durable text workflow components

A durable run checks its provider and components each time `plan()`,
`execute()` or `resume()` starts, before routing or planning. A rejected
`run_store`, `verifier`, `memory`, `tool_runtime` or `checkpoint_store`
raises `WorkflowBindingError`, a `ValueError` whose message starts with the
argument's name.

| Component | Accepted | Rejected |
|---|---|---|
| `run_store` | `SQLiteWorkflowStore` | Subclasses |
| `verifier` | `None`, `TokenVerifier`, or a validator declared with `LocalOnly(..., role="validator")` | `CallableVerifier`, other verifiers, `TokenVerifier` subclasses and other `LocalOnly` roles |
| `acceptance` | `None`, an advisory policy, or a strict `AcceptancePolicy` whose final validators are registered validator objects | A value that is not an `AcceptancePolicy`; a strict policy without final validators under a synthesis other than `DELIVERABLE` |
| `memory` | `None` | Any planner memory |
| `tool_runtime` | `None`; nodes send plain text | Any tool runtime |
| `read_tools` | `ReadProfile` declarations, named by read steps in graphs and templates you write | Other objects, duplicate names and `read_tools` without `run_store`; read steps naming an undeclared profile, with invalid input or judging another node; read steps on ordinary runs |
| `checkpoint_store` | `None`; the journal holds checkpoints | Any checkpoint store |
| Architects and other components | Built-in components and `LocalOnly` factories | Subclasses and unjournaled dependencies |
| Node inputs | Plain text | `attach_dep_artifacts`, `attachments` and `artifacts` |
| Resume | The saved recipe: components, models, budget, `retry_backoff_s`, concurrency, gate validator, acceptance policy and read profiles | Any change raises `WorkflowConflictError` |
| `max_budget_usd=None` | Runs without a budget | |
| `artifact_dir`, `max_tool_iterations` | Ignored | |

A caller-built graph with image inputs fails before any node runs. A
generated plan with them goes back to the planner for a repair, which is a
new paid planning call.

Tests:

- Rejected components: `tests/test_support_matrix.py::test_durable_run_rejects_component_before_any_call[store-subclass]`,
  `tests/test_support_matrix.py::test_durable_run_rejects_component_before_any_call[callable-verifier]`,
  `tests/test_support_matrix.py::test_durable_run_rejects_component_before_any_call[token-verifier-subclass]`,
  `tests/test_support_matrix.py::test_durable_run_rejects_component_before_any_call[memory]`,
  `tests/test_support_matrix.py::test_durable_run_rejects_component_before_any_call[tool-runtime]`,
  `tests/test_support_matrix.py::test_durable_run_rejects_component_before_any_call[checkpoint-store]`,
  `tests/test_workflow_acceptance.py::test_durable_runs_refuse_unregistered_verifier_code_before_any_call`.
- Gate validators: `tests/test_workflow_acceptance.py::test_registered_validator_judges_durable_gates_from_the_exact_target_text`,
  `tests/test_workflow_acceptance.py::test_a_changed_validator_configuration_or_version_refuses_resume_before_any_call`,
  `tests/test_workflow_acceptance.py::test_a_validator_that_cannot_decide_ends_its_gate_without_regeneration`,
  `tests/test_workflow_binding.py::test_local_only_validator_refuses_a_provider_coroutine_or_awaitable_factory`,
  `tests/test_workflow_binding.py::test_local_only_validator_refuses_what_its_factory_builds`.
- Strict acceptance: `tests/test_workflow_acceptance.py::test_strict_durable_run_decides_on_the_delivered_output`,
  `tests/test_workflow_acceptance.py::test_strict_durable_gate_reads_and_records_its_final_verdict`,
  `tests/test_workflow_acceptance.py::test_callable_verifier_stays_rejected_on_durable_runs`,
  `tests/test_workflow_acceptance.py::test_the_durable_runtime_checks_its_policy_itself`
  and `tests/test_workflow_acceptance.py::test_resume_refuses_another_acceptance_policy_before_any_call`;
  generated gates and obligations `tests/test_workflow_acceptance.py::test_a_durable_gate_from_a_generated_plan_does_not_accept_on_its_own`,
  `tests/test_workflow_acceptance.py::test_a_durable_strict_run_refuses_a_broken_gate_before_any_call`,
  `tests/test_workflow_acceptance.py::test_a_durable_strict_run_refuses_output_no_gate_could_accept`
  and `tests/test_workflow_acceptance.py::test_a_durable_step_added_after_a_gate_is_refused_without_a_final_validator`;
  recovery `tests/test_workflow_acceptance.py::test_crash_after_synthesis_before_the_decision_decides_again`,
  `tests/test_workflow_acceptance.py::test_corrupted_stored_output_is_not_accepted`,
  `tests/test_workflow_process_loss.py::test_killed_durable_run_resumes_its_decision_without_new_calls`
  and `tests/test_workflow_process_loss.py::test_killed_priced_run_keeps_its_charges_generations_and_regeneration`.
- Read steps: `tests/test_workflow_read_tool.py::test_a_read_step_runs_without_a_model_call`,
  `tests/test_workflow_read_tool.py::test_only_developer_graphs_declare_read_steps`,
  `tests/test_workflow_read_tool.py::test_recipes_record_read_tools_only_when_declared`
  and `tests/test_workflow_read_tool.py::test_source_content_never_changes_what_a_run_may_do`;
  limits `tests/test_workflow_read_tool.py::test_private_destinations_are_refused_before_anything_is_sent`,
  `tests/test_workflow_read_tool.py::test_reads_are_limited_in_bytes_redirects_and_destinations`,
  `tests/test_workflow_read_tool.py::test_reads_are_limited_in_time`,
  `tests/test_workflow_read_tool.py::test_a_slow_source_cannot_hold_a_read_past_its_time_limit`
  and `tests/test_workflow_read_tool.py::test_a_step_that_stops_waiting_stops_its_read`; recovery
  `tests/test_workflow_read_tool.py::test_a_saved_read_response_is_used_on_resume_without_reading_again`,
  `tests/test_workflow_read_tool.py::test_a_lost_read_response_is_read_again_only_when_free_and_repeat_safe`
  and `tests/test_workflow_read_tool.py::test_killed_read_resumes_without_a_duplicate_read`.
- Plain-text nodes: `tests/test_support_matrix.py::test_durable_run_rejects_image_inputs_on_a_caller_graph_before_any_call`,
  `tests/test_workflow_native_runtime.py::test_plan_the_run_cannot_execute_is_repaired_as_an_additional_paid_attempt`
  and `tests/test_workflow_provider.py::test_unsupported_input_rejected_before_journal_or_http`.
- Built-in components: `tests/test_workflow_binding.py::test_local_only_factory_creates_independent_local_architects`,
  `tests/test_workflow_binding.py::test_custom_subclass_cannot_silently_inherit_builtin_binding`,
  `tests/test_workflow_binding.py::test_preflight_rejects_unjournaled_dependencies`.
- Resume: `tests/test_support_matrix.py::test_durable_run_rejects_a_changed_retry_backoff_before_any_call`
  and `tests/test_workflow_runtime.py::test_changed_handoff_binding_rejected_before_calls`.
- No budget and ignored options: `tests/test_support_matrix.py::test_durable_run_ignores_artifact_dir_and_tool_iterations_and_allows_no_budget`.

## Jobs

`jobs validate`, `jobs plan` and `jobs run` check every operation against its
profile before a run is created and before any provider call.

| Setting | Accepted | Rejected |
|---|---|---|
| Profile `provider` | `offline`, `openai_image`, `gemini_image` | Any other name |
| `offline` artifact | `image/png`, with no dimensions or 1x1 | Any other MIME type or size |
| `openai_image` artifact | The MIME type of `output_format`; the dimensions of `size` unless it is `auto` | Mismatches, attachments and `base_url` |
| `gemini_image` artifact | PNG, JPEG, GIF or WebP; reference images | Other MIME types; `response_modalities` without `IMAGE` |
| Validators | Names and versions in an operation's `artifact.validators`, registered in-process with `preflight_job` and `JobRunner` | Unregistered names or versions, before any run; validator fields elsewhere; the CLI registers none |
| Tools, memory | None | Manifest fields for them are unknown fields |

A Gemini image's dimensions are checked when it arrives; a mismatch rejects
that attempt.

Tests:

- Accepted profiles: `tests/test_jobs_runtime_support.py::test_offline_profile_accepts_a_png_with_no_or_1x1_dimensions`,
  `tests/test_jobs_runtime_support.py::test_provider_preflight_accepts_complete_local_openai_setup`,
  `tests/test_jobs_runtime_support.py::test_gemini_image_profile_accepts_raster_artifacts_and_reference_images`.
- Artifacts a profile cannot produce, with no provider call and no run:
  `tests/test_support_matrix.py::test_jobs_rejects_an_artifact_its_profile_cannot_produce_before_any_call[offline-jpeg]`,
  `tests/test_support_matrix.py::test_jobs_rejects_an_artifact_its_profile_cannot_produce_before_any_call[offline-1024]`,
  `tests/test_support_matrix.py::test_jobs_rejects_an_artifact_its_profile_cannot_produce_before_any_call[gemini-pdf]`,
  `tests/test_support_matrix.py::test_jobs_rejects_an_artifact_its_profile_cannot_produce_before_any_call[openai-format]`,
  `tests/test_jobs_runtime_support.py::test_offline_profile_rejects_an_artifact_other_than_a_1x1_png`,
  `tests/test_jobs_runtime_support.py::test_gemini_image_profile_rejects_an_artifact_it_cannot_return`,
  `tests/test_jobs_cli.py::test_artifact_the_profile_cannot_produce_fails_preflight_before_any_run`.
- Other refusals: `tests/test_support_matrix.py::test_jobs_manifest_accepts_only_image_provider_kinds`,
  `tests/test_support_matrix.py::test_jobs_manifest_has_no_validator_tool_or_memory_fields`,
  `tests/test_jobs_runtime_support.py::test_openai_reference_inputs_fail_before_provider_construction`,
  `tests/test_jobs_runtime_support.py::test_job_manifest_cannot_redirect_openai_credentials`,
  `tests/test_jobs_runtime_support.py::test_gemini_image_profile_requires_image_modality`.
- Registered validators: `tests/test_jobs_preflight.py::test_validators_a_manifest_names_must_be_registered_before_any_run`,
  `tests/test_jobs_runner.py::test_runs_refuse_validators_the_runner_has_not_registered`,
  `tests/test_jobs_runner.py::test_registered_validator_checks_the_published_bytes_and_is_recorded`,
  `tests/test_jobs_runner.py::test_validator_errors_are_indeterminate_and_never_unknown_outcomes`.

[Documentation map](index.md) · [Architecture](architecture.md) ·
[Roadmap](../ROADMAP.md)
