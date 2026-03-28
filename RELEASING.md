# VelociRAG Release Process

## Prerequisites

- All changes committed and pushed to `main`
- `~/.local/bin/mcp-publisher` installed ([GitHub releases](https://github.com/modelcontextprotocol/registry/releases))
- PyPI trusted publishing configured (GitHub Actions)
- AUR SSH key in GitHub secrets (`AUR_SSH_PRIVATE_KEY`)

## Release Flow

### 1. Bump version

```bash
cd ~/velocirag
# Update version in pyproject.toml
sed -i 's/version = "OLD"/version = "NEW"/' pyproject.toml
# Update version in server.json (both fields)
sed -i 's/"version": "OLD"/"version": "NEW"/g' server.json
# Commit
git add -A && git commit -m "vNEW: <release title>" && git push
```

### 2. Wait for CI

```bash
# Watch CI pass before tagging
gh run watch --repo HaseebKhalid1507/VelociRAG
```

CI must be ✅ green. Do not tag on red.

### 3. Tag and release

```bash
git tag vNEW && git push origin vNEW

gh release create vNEW \
  --repo HaseebKhalid1507/VelociRAG \
  --title "vNEW — <release title>" \
  --latest \
  --notes "<release notes>"
```

This triggers the `publish.yml` workflow which handles:
- **PyPI** — builds wheel + sdist, uploads with trusted publishing
- **AUR** — updates PKGBUILD, pushes to AUR

### 4. Verify publish

```bash
gh run watch --repo HaseebKhalid1507/VelociRAG
# Both "Publish to PyPI" jobs should be ✅
```

### 5. Update MCP Registry

```bash
cd ~/velocirag
~/.local/bin/mcp-publisher login github
~/.local/bin/mcp-publisher publish
```

Requires browser for GitHub OAuth. Only manual step.

### 6. Update production (Jade)

```bash
cd ~/velocirag && git pull && source venv/bin/activate && pip install -e .
# Reindex if chunking/embedding changes
systemctl --user restart jawz-search-daemon
```

## Automated by CI

| Target | How | Trigger |
|--------|-----|---------|
| PyPI | `pypa/gh-action-pypi-publish` with `skip-existing: true` | `release: published` |
| AUR | `KSXGitHub/github-actions-deploy-aur` | `release: published` (after PyPI) |

## Manual Steps

| Target | How | When |
|--------|-----|------|
| MCP Registry | `mcp-publisher login github && mcp-publisher publish` | After PyPI is live |
| Jade production | `pip install -e .` + restart daemon | After release |

## Quick Release Script

```bash
#!/bin/bash
# Usage: ./release.sh 0.7.2 "Feature description"
VERSION=$1
TITLE=$2

if [ -z "$VERSION" ] || [ -z "$TITLE" ]; then
    echo "Usage: ./release.sh <version> <title>"
    exit 1
fi

cd ~/velocirag

# Bump
sed -i "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml
sed -i "s/\"version\": \".*\"/\"version\": \"$VERSION\"/g" server.json
git add -A && git commit -m "v${VERSION}: ${TITLE}" && git push

# Wait for CI
echo "Waiting for CI..."
gh run watch --repo HaseebKhalid1507/VelociRAG

# Tag + release
git tag "v${VERSION}" && git push origin "v${VERSION}"
gh release create "v${VERSION}" \
    --repo HaseebKhalid1507/VelociRAG \
    --title "v${VERSION} — ${TITLE}" \
    --latest \
    --notes "$(git log --oneline $(git describe --tags --abbrev=0 HEAD~1)..HEAD~1)"

# Wait for publish
echo "Waiting for PyPI publish..."
gh run watch --repo HaseebKhalid1507/VelociRAG

# MCP registry
echo "Publishing to MCP registry..."
~/.local/bin/mcp-publisher login github
~/.local/bin/mcp-publisher publish

# Update local
pip install -e .
echo "✅ v${VERSION} shipped everywhere"
```
