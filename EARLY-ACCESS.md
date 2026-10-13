# TandemLLM early access

You have read access to this repository. It holds the releases only, one branch, `main`, with a tag for each release.

## Install on a DGX Spark

The repository is private, so clone it with your own GitHub login. Use SSH if your key is on GitHub:

```bash
git clone git@github.com:0xBakeer/TandemLLM-early-access.git
bash TandemLLM-early-access/install.sh --setup --yes --repo git@github.com:0xBakeer/TandemLLM-early-access.git
```

Or use HTTPS through the GitHub CLI, after `gh auth login` and `gh auth setup-git`:

```bash
gh repo clone 0xBakeer/TandemLLM-early-access
bash TandemLLM-early-access/install.sh --setup --yes --repo https://github.com/0xBakeer/TandemLLM-early-access.git
```

Then pick a model. The first command shows what it would download; the second downloads it, checks every file and starts the engine on port 8000:

```bash
~/TandemLLM/run/ops/use-profile.sh flashnext-3.05 --dry-run
~/TandemLLM/run/ops/use-profile.sh flashnext-3.05
```

`flashnext-3.05` downloads 85.1 GB and needs about 145 GB free on disk. For Qwen3.8-27B use `exl3-4.5` (about 58 GB, 118 GB free); `use-profile.sh --list` prints every profile. No Hugging Face token is needed. If a download stops, run the same command again. It keeps the files that finished.

Open the dashboard at `http://<your-spark>:8000/dashboard/`. The Chat tab is in beta; its known limitations are in [docs/chat.md](docs/chat.md) and `CHANGELOG.md`.

## The manual

Read the book as a PDF: [manual/tandemllm-manual.pdf](manual/tandemllm-manual.pdf). You need no LaTeX for it. Chapter 3, "Installing TandemLLM", repeats these steps in its section "Installing from the early-access repository", and explains what each download is for. Every weight file is public on Hugging Face; the README's model table lists each profile with its size.

## Updating

Once a profile runs, the box counts as configured, and `--setup` alone refuses to touch it. Add `--force-setup`: the installer stops the engine, checks out the new tag and comes back in setup mode. Then switch to your profile again. Its files are already on disk, so nothing downloads twice:

```bash
bash ~/TandemLLM/src/install.sh --setup --force-setup --yes --ref <new tag> --repo git@github.com:0xBakeer/TandemLLM-early-access.git
~/TandemLLM/run/ops/use-profile.sh flashnext-3.05
```
