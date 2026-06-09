#!/bin/bash
# install_mcp.sh — Register mapgen as an MCP server in Claude Desktop
# Run once after installing mapgen. Re-run to update the path.

set -e

MAPGEN_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$MAPGEN_DIR/.venv/bin/python"
MCP_SCRIPT="$MAPGEN_DIR/mapgen_mcp.py"
CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"

echo "📍 mapgen dir:    $MAPGEN_DIR"
echo "🐍 Python:        $VENV_PYTHON"
echo "⚙️  Claude config: $CLAUDE_CONFIG"
echo ""

# ── Check venv exists ────────────────────────────────────────────────────────
if [ ! -f "$VENV_PYTHON" ]; then
    echo "🔧 Creating virtual environment and installing dependencies..."
    cd "$MAPGEN_DIR"
    uv venv .venv --python 3.11
    uv pip install mcp geopandas pandas openpyxl rapidfuzz mapclassify folium matplotlib pyogrio
    echo "✅ Dependencies installed."
fi

# ── Patch claude_desktop_config.json ────────────────────────────────────────
mkdir -p "$(dirname "$CLAUDE_CONFIG")"

if [ ! -f "$CLAUDE_CONFIG" ]; then
    echo "{}" > "$CLAUDE_CONFIG"
fi

# Use Python to safely merge the mcpServers entry
"$VENV_PYTHON" - <<PYEOF
import json, sys
from pathlib import Path

config_path = Path("$CLAUDE_CONFIG")
config = json.loads(config_path.read_text()) if config_path.exists() else {}

config.setdefault("mcpServers", {})
config["mcpServers"]["mapgen"] = {
    "command": "$VENV_PYTHON",
    "args": ["$MCP_SCRIPT"]
}

config_path.write_text(json.dumps(config, indent=2))
print("✅ Claude Desktop config updated.")
PYEOF

echo ""
echo "🗺️  mapgen MCP server registered!"
echo ""
echo "Next steps:"
echo "  1. Quit Claude Desktop completely (Cmd+Q)"
echo "  2. Reopen Claude Desktop"
echo "  3. Look for the 🔧 tools icon — mapgen tools will appear"
echo "  4. Try: 'Map this data' and attach your CSV/Excel file"
echo ""
