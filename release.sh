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
sed -i "0,/^version = \".*\"/s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
sed -i "s/\"version\": \".*\"/\"version\": \"$VERSION\"/g" server.json
git add -A && git commit -m "v${VERSION}: ${TITLE}" && git push

# Wait for CI
echo "⏳ Waiting for CI..."
gh run watch --repo HaseebKhalid1507/VelociRAG

# Tag + release
git tag "v${VERSION}" && git push origin "v${VERSION}"
gh release create "v${VERSION}" \
    --repo HaseebKhalid1507/VelociRAG \
    --title "v${VERSION} — ${TITLE}" \
    --latest \
    --notes "$(git log --oneline $(git describe --tags --abbrev=0 HEAD~1 2>/dev/null || echo HEAD~5)..HEAD~1)"

# Wait for publish
echo "⏳ Waiting for PyPI + AUR publish..."
sleep 10
gh run watch --repo HaseebKhalid1507/VelociRAG

# MCP registry
echo "📦 Publishing to MCP registry..."
~/.local/bin/mcp-publisher login github
~/.local/bin/mcp-publisher publish

# Update local
source ~/velocirag/venv/bin/activate
pip install -e .

echo "✅ v${VERSION} — ${TITLE} — shipped everywhere"
