# docs/plan/ — Orchestration Plan Hygiene

When executing a plan file in this directory:

- Update task sections immediately after completion:
  - `status`
  - `log`
  - `files edited/created`
- Keep task logs concise and implementation-specific.
- Ensure each completed task corresponds to committed code.
- Prefer one commit per completed task/wave when practical.
- Do not mark tasks complete unless validation for that task passed.
