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
| `OfflineProvider` | Yes | No | Acknowledged in the text only | 1x1 PNGs with `artifacts_per_call` |
| `AnthropicProvider` | Yes | Yes | Yes | No |
| `OpenAIProvider` | Yes | Yes, non-strict | Yes, as data URLs | No |
| `OpenAIImageProvider` | Prompts only | Rejected | Rejected | Yes |
| `GeminiProvider` | Yes | Yes | Yes | Image models, or explicit `response_modalities` |
| `OpenAIResponsesProvider` | `gpt-6-astra`, `gpt-5.6-sol` | Yes, non-strict | Rejected | No |
| `AnthropicMessagesProvider` | `claude-fable-5-1` | Rejected | Rejected | No |
| Custom `Provider` | Yes, through `complete()` | Only if it overrides `chat()` | Only if it overrides `chat()` | If it returns `artifacts` |

No provider supports structured output or streaming. Planners ask for JSON
and parse it from the reply text. Test:
`tests/test_support_matrix.py::test_no_provider_offers_structured_output_or_streaming`.

Ordinary runs check these features when a node calls its provider, after
planning, so a model-generated plan has already been paid for. "Rejected"
means that call raises `ValueError` before sending a request. A provider
without tool support raises `NotImplementedError` for a node with tools.

Tests:

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
| `OfflineProvider` | Static plan and echo only | `offline` | Ordinary: blended token estimate. Durable and Jobs: zero | None |
| `AnthropicProvider` | Rejected | None | Blended token estimate | The Anthropic SDK default |
| `OpenAIProvider` | Rejected | None | Blended token estimate | The OpenAI SDK default, unless `max_retries` is set |
| `OpenAIImageProvider` | Rejected | `openai_image` | Per-call ceiling; without one, a per-image estimate marked incomplete, or unknown | The OpenAI SDK default, unless `max_retries` is set. Jobs sets 0 |
| `GeminiProvider` | Rejected | `gemini_image` | Text: blended token estimate. Images: as for `OpenAIImageProvider` | None configured |
| `OpenAIResponsesProvider` | `gpt-6-astra`, `gpt-5.6-sol` | None | Exact native price | None; a client with retries is refused |
| `AnthropicMessagesProvider` | `claude-fable-5-1` | None | Exact native price | None; a client with retries is refused |
| Custom `Provider` | Rejected | None | Blended token estimate, or its own `cost_usd` | Its own |

Each path keeps money in its own unit:

- **Ordinary runs** record float USD in `Sentinel`. The ledger covers execution
  and synthesis (`cost_scope="execution_and_synthesis"`); routing, planning
  and successful supervision calls are outside it. With `max_budget_usd`
  set, `OpenAIImageProvider`, `OpenAIResponsesProvider`,
  `AnthropicMessagesProvider` and image-model `GeminiProvider` calls need
  `max_cost_per_call_usd`; without it the node raises
  `BudgetEstimateRequired` before its call. See [cost guardrails](budgets.md).
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
- Price ceilings for capped runs: `tests/test_provider.py::test_openai_image_hard_budget_requires_inclusive_call_ceiling`,
  `tests/test_provider.py::test_gemini_hard_budget_uses_only_inclusive_call_ceiling`,
  `tests/test_provider_responses.py::test_legacy_budget_requires_inclusive_explicit_ceiling`.
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
| `verifier` | `None` or `TokenVerifier` | `CallableVerifier`, other verifiers and `TokenVerifier` subclasses |
| `memory` | `None` | Any planner memory |
| `tool_runtime` | `None`; nodes send plain text | Any tool runtime |
| `checkpoint_store` | `None`; the journal holds checkpoints | Any checkpoint store |
| Architects and other components | Built-in components and `LocalOnly` factories | Subclasses and unjournaled dependencies |
| Node inputs | Plain text | `attach_dep_artifacts`, `attachments` and `artifacts` |
| Resume | The saved recipe: components, models, budget, `retry_backoff_s` and concurrency | Any change raises `WorkflowConflictError` |
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
  `tests/test_support_matrix.py::test_durable_run_rejects_component_before_any_call[checkpoint-store]`.
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
| Validators, tools, memory | None | Manifest fields for them are unknown fields |

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

[Documentation map](index.md) · [Architecture](architecture.md) ·
[Roadmap](../ROADMAP.md)
