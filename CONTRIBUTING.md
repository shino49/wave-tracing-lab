# Development workflow

Use `main` for reviewed, tested milestones and `dot/dev` for assistant development.
Keep completed development changes synchronized to `dot/dev`; group coherent,
validated changes into pull requests before merging into `main`. Do not rewrite
published history or force-push without explicit approval.

Run the suites relevant to a change and record their actual results. Distinguish
CPU checks, DOM simulations, native software WebGPU execution, and physical
browser/GPU testing. A skipped or unavailable adapter is not a passing GPU test.
Preserve model limitations and diagnostics rather than hiding unsupported paths.

Before publishing, review the diff for secrets, personal data, machine paths,
deployment identities, generated renders, logs, and third-party license notices.
Do not add credentials, `.env` files, `.openai` state, `node_modules`, or private
history. Retain attribution and the component-specific licenses described in
README.md. Selected numerical fixtures are intentional source inputs; review
fixture changes separately from generated test reports.

Run `npm ci` to install pinned development dependencies. Python reference tests
use requirements-test.txt in a virtual environment. Serve `dist/` over localhost
or HTTPS; browser modules are editable sources and require no compilation.

The branch convention describes development sessions and milestone reviews.
It does not install a background filesystem watcher or GitHub Actions workflow.
