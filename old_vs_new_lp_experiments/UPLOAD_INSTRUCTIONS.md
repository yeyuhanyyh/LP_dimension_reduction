# Upload Instructions

The current machine does not expose a local `git` or `gh` executable, and this
folder is not inside a git repository.  The prepared archive is:

`github_upload_old_vs_new_lp_experiments_20260503_175604.zip`

Recommended GitHub upload options:

1. Create or open a GitHub repository.
2. Upload either this whole folder or the zip file through the GitHub web UI.
3. If using git on another machine:

```bash
git clone <repo-url>
cp -R github_upload_old_vs_new_lp_experiments_20260503_175604 <repo>/
cd <repo>
git add github_upload_old_vs_new_lp_experiments_20260503_175604
git commit -m "Add old and new LP experiment artifacts"
git push
```

If a target repository name is provided in `owner/repo` form and binary upload
credentials are available, this bundle can be pushed there directly.
