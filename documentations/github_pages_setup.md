# Enabling GitHub Pages for `https://mattral.github.io/KANX/`

You reported that `https://mattral.github.io/KANX/architecture/` shows a 404.
That's because GitHub Pages hasn't been wired up to the `gh-pages` branch
yet for this repo. Here is **exactly** what to do — one-time setup, then
every `git push origin main` auto-deploys the MkDocs Material site.

## Step 1 — Push the latest code to `main`

```bash
git add -A
git commit -m "feat(v0.1.6): docs site + grid-range guard + honest benchmark + API hardening"
git push origin main
```

This triggers `.github/workflows/docs.yml`, which:

1. Builds `mkdocs build --strict` (fails on broken links).
2. Runs `mkdocs gh-deploy --force --clean` — this creates / updates the
   `gh-pages` branch on your repo containing the rendered HTML.

You can also trigger it manually from the **Actions tab → Docs → Run workflow**.

## Step 2 — Enable Pages in repo Settings

After the first workflow run completes (~1 minute), go to:

**Repo → Settings → Pages**

Configure:

- **Source**: `Deploy from a branch`
- **Branch**: `gh-pages` &nbsp;·&nbsp; **Folder**: `/ (root)`
- Click **Save**.

GitHub will print a banner: *"Your site is live at https://mattral.github.io/KANX/"*

> Alternatively, set **Source** = `GitHub Actions` and replace `docs.yml` with
> the official `actions/upload-pages-artifact` + `actions/deploy-pages` flow.
> Either path works; `gh-deploy` (the one we use) is simpler and is what
> MkDocs documents officially.

## Step 3 — Verify

Open these URLs (they should all return HTTP 200 after the first deploy):

- <https://mattral.github.io/KANX/>                      &nbsp;·&nbsp; landing page
- <https://mattral.github.io/KANX/quickstart/>           &nbsp;·&nbsp; quickstart
- <https://mattral.github.io/KANX/architecture/>         &nbsp;·&nbsp; the one you reported as 404
- <https://mattral.github.io/KANX/benchmarks/>           &nbsp;·&nbsp; fair benchmark
- <https://mattral.github.io/KANX/api/>                  &nbsp;·&nbsp; REST API reference

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 404 on every page | Pages source not set | Settings → Pages → branch = `gh-pages` |
| 404 only on `/architecture/` etc. | First deploy not run yet | Check Actions tab → Docs workflow |
| Docs deploy fails with `strict` error | A markdown link in `docs/` is broken | `mkdocs build --strict` locally to see the file:line |
| `gh-pages` branch never created | `permissions: contents: write` missing | Already set in `docs.yml` |
| Site stuck on old content | Browser cache | Hard-refresh (Cmd-Shift-R) or `?v=1` query string |

## Local preview

```bash
pip install mkdocs mkdocs-material
mkdocs serve     # http://127.0.0.1:8000
```

This is the exact same site that gets deployed — useful for verifying a PR
before merging.
