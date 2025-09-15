#!/bin/zsh
set -euo pipefail

# --- Config ---
PKG_DIR="src/arbeitsplaner"
PKG_NAME="arbeitsplaner"
OUT_DIR="diagrams"
JSON_OUT="$OUT_DIR/pycg_callgraph.json"
DOT_OUT="$OUT_DIR/pycg_callgraph.dot"
SVG_OUT="$OUT_DIR/pycg_callgraph.svg"
PYCG_PY="${PYCG_PY:-$(command -v python)}"   # Interpreter für PyCG (z.B. .pycg310/bin/python)
FILTER_QT="${FILTER_QT:-0}"                  # 1 = Qt-Knoten rausfiltern
RANKDIR="${RANKDIR:-LR}"                     # LR oder TB

# --- Checks ---
mkdir -p "$OUT_DIR"
if [ ! -d "$PKG_DIR" ]; then
  echo "❌ Paketordner nicht gefunden: $PKG_DIR"
  exit 1
fi

# Prüfen, ob pycg-Modul vorhanden ist
"$PYCG_PY" - <<'PY'
import importlib.util as u, sys
ok = bool(u.find_spec("pycg") and u.find_spec("pycg.__main__"))
print("OK" if ok else "NO")
PY
if [ "$(tail -n1 <<<"$( "$PYCG_PY" - <<'PY'
import importlib.util as u, sys
ok = bool(u.find_spec("pycg") and u.find_spec("pycg.__main__"))
print("OK" if ok else "NO")
PY
)")" != "OK" ]; then
  echo "❌ pycg.__main__ nicht importierbar mit $PYCG_PY"
  echo "   Tipp: PYCG_PY=/Pfad/zu/.pycg310/bin/python scripts/make_callgraphs.sh"
  exit 1
fi

# Graphviz (dot) prüfen
if ! command -v dot >/dev/null 2>&1; then
  echo "❌ Graphviz 'dot' nicht gefunden. Installiere z.B.: brew install graphviz"
  exit 1
fi

# --- 1) Alle .py-Dateien einsammeln ---
FILELIST="$OUT_DIR/_pycg_files.txt"
find "$PKG_DIR" -type f -name "*.py" | sort > "$FILELIST"
COUNT=$(wc -l < "$FILELIST" | tr -d ' ')
if [ "$COUNT" = "0" ]; then
  echo "❌ Keine .py-Dateien in $PKG_DIR gefunden."
  exit 1
fi
echo "📄 $COUNT Dateien gefunden"

# --- 2) PyCG ausführen (JSON-Ausgabe) ---
echo "🚀 PyCG läuft (JSON) ..."
# shellcheck disable=SC2046
"$PYCG_PY" -m pycg.__main__ $(cat "$FILELIST") --package "$PKG_NAME" -o "$JSON_OUT"

# Validierung: JSON muss kein leeres Objekt sein
if [ "$(wc -c < "$JSON_OUT")" -lt 5 ]; then
  echo "⚠️ Ergebnis sehr klein. Inhalt:"
  cat "$JSON_OUT"
fi

# --- 3) JSON -> DOT konvertieren ---
echo "🔄 JSON → DOT"
FILTER_ARGS=()
if [ "$FILTER_QT" = "1" ]; then
  # Qt-Knoten rausfiltern (PyQt/PySide)
  FILTER_ARGS+=( --filter '^PyQt6\.' --filter '^PySide6\.' )
fi

PY="$(command -v python || true)"
if [ -z "$PY" ]; then PY="$PYCG_PY"; fi

"$PY" tools/json2dot.py "$JSON_OUT" "$DOT_OUT" "${FILTER_ARGS[@]}" --rankdir "$RANKDIR"

# --- 4) DOT -> SVG rendern ---
echo "🖼  DOT → SVG"
dot -Tsvg "$DOT_OUT" -o "$SVG_OUT"

echo "✅ Fertig:"
echo "   JSON: $JSON_OUT"
echo "   DOT : $DOT_OUT"
echo "   SVG : $SVG_OUT"