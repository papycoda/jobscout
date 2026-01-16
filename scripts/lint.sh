#!/bin/bash
# Simple lint using Python's built-in tools

echo "Checking Python files..."

FAILED=0

for file in backend/*.py jobscout/*.py; do
    if [ -f "$file" ]; then
        # Try to compile each file
        if ! python -m py_compile "$file" 2>&1; then
            echo "❌ $file has syntax errors"
            FAILED=1
        fi
    fi
done

# Try to import main modules
echo ""
echo "Testing imports..."
if ! python -c "from backend.main import app" 2>&1 | grep -q "Error"; then
    echo "✓ backend.main imports OK"
else
    echo "❌ backend.main has import errors"
    FAILED=1
fi

if ! python -c "from backend.metrics import metrics_tracker" 2>&1 | grep -q "Error"; then
    echo "✓ backend.metrics imports OK"
else
    echo "❌ backend.metrics has import errors"
    FAILED=1
fi

if ! python -c "from backend.llm import extract_profile" 2>&1 | grep -q "Error"; then
    echo "✓ backend.llm imports OK"
else
    echo "❌ backend.llm has import errors"
    FAILED=1
fi

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "✅ All checks passed"
    exit 0
else
    echo ""
    echo "❌ Some checks failed"
    exit 1
fi
