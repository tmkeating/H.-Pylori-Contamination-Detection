#!/bin/bash
# verify_venv.sh - Centralized Virtual Environment Verification
#
# Purpose: Verify that the Python virtual environment is properly configured
# with all required packages from requirements.txt
#
# Usage:
#   source ./verify_venv.sh
#   OR
#   bash ./verify_venv.sh
#
# Sets: VENV_ROOT (global) for use in calling scripts
# Exits with error (1) if venv is missing or packages are incomplete

set -e

verify_venv() {
    echo "=========================================================================="
    echo "Verifying Python Virtual Environment..."
    echo "=========================================================================="
    echo ""
    
    # Get VENV_ROOT from config
    VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)" 2>/dev/null)
    
    if [ -z "$VENV_ROOT" ]; then
        echo "❌ ERROR: Could not determine VENV_ROOT from config.py"
        return 1
    fi
    
    echo "Expected venv location: $VENV_ROOT"
    
    # Check if venv directory exists
    if [ ! -d "$VENV_ROOT" ]; then
        echo "❌ ERROR: Virtual environment directory not found at $VENV_ROOT"
        echo ""
        echo "To create it, run:"
        echo "  python3 -m venv $VENV_ROOT"
        echo "  source $VENV_ROOT/bin/activate"
        echo "  pip install --upgrade pip"
        echo "  pip install -r requirements.txt"
        return 1
    fi
    
    # Check if activation script exists
    if [ ! -f "$VENV_ROOT/bin/activate" ]; then
        echo "❌ ERROR: Virtual environment activation script not found"
        echo "  Expected: $VENV_ROOT/bin/activate"
        return 1
    fi
    
    # Test venv activation and verify all packages from requirements.txt
    echo "Testing virtual environment activation..."
    
    # Read requirements.txt and check each package
    if [ ! -f "requirements.txt" ]; then
        echo "❌ ERROR: requirements.txt not found in current directory"
        return 1
    fi
    
    # Generate Python code to check all packages
    PACKAGE_CHECK=$(bash -c "source $VENV_ROOT/bin/activate && python3 << 'PYEOF'
import sys

# Read requirements.txt
packages_to_check = []
try:
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Handle version specifiers (e.g., torch==2.7.1)
                pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].split('!=')[0].strip()
                packages_to_check.append(pkg_name)
except Exception as e:
    print(f'ERROR: Could not read requirements.txt: {e}')
    sys.exit(1)

# Check each package
missing_packages = []
for pkg in packages_to_check:
    try:
        # Map some package names to their import names
        import_name = pkg
        if pkg == 'scikit-learn':
            import_name = 'sklearn'
        elif pkg == 'pillow':
            import_name = 'PIL'
        
        __import__(import_name)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'❌ {pkg}: MISSING')
        missing_packages.append(pkg)

# Exit with error if any packages are missing
if missing_packages:
    print(f'ERROR: Missing packages: {missing_packages}')
    sys.exit(1)
PYEOF
" 2>&1)
    
    # Print the package check output
    echo ""
    echo "Checking required packages from requirements.txt:"
    echo "$PACKAGE_CHECK" | while read line; do
        echo "  $line"
    done
    
    # Check if any packages were missing
    if echo "$PACKAGE_CHECK" | grep -q "ERROR"; then
        echo ""
        echo "⚠️  WARNING: Required packages are missing from the virtual environment"
        echo ""
        echo "Installing missing packages automatically..."
        echo ""
        
        # Activate venv and install missing packages
        bash -c "source $VENV_ROOT/bin/activate && pip install -r requirements.txt" || {
            echo ""
            echo "❌ ERROR: Failed to install missing packages"
            echo ""
            echo "To manually install, run:"
            echo "  source $VENV_ROOT/bin/activate"
            echo "  pip install -r requirements.txt"
            echo ""
            return 1
        }
        
        echo ""
        echo "Re-verifying packages after installation..."
        echo ""
        
        # Re-check packages after installation
        PACKAGE_CHECK=$(bash -c "source $VENV_ROOT/bin/activate && python3 << 'PYEOF'
import sys

# Read requirements.txt
packages_to_check = []
try:
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Handle version specifiers (e.g., torch==2.7.1)
                pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].split('!=')[0].strip()
                packages_to_check.append(pkg_name)
except Exception as e:
    print(f'ERROR: Could not read requirements.txt: {e}')
    sys.exit(1)

# Check each package
missing_packages = []
for pkg in packages_to_check:
    try:
        # Map some package names to their import names
        import_name = pkg
        if pkg == 'scikit-learn':
            import_name = 'sklearn'
        elif pkg == 'pillow':
            import_name = 'PIL'
        
        __import__(import_name)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'❌ {pkg}: MISSING')
        missing_packages.append(pkg)

# Exit with error if any packages are missing
if missing_packages:
    print(f'ERROR: Missing packages: {missing_packages}')
    sys.exit(1)
PYEOF
" 2>&1)
        
        # Print the re-check output
        echo "Verified packages after installation:"
        echo "$PACKAGE_CHECK" | while read line; do
            echo "  $line"
        done
        
        # Check again if any packages are still missing
        if echo "$PACKAGE_CHECK" | grep -q "ERROR"; then
            echo ""
            echo "❌ ERROR: Some packages failed to install"
            echo ""
            return 1
        fi
    fi
    
    echo ""
    echo "✓ Virtual environment verified successfully"
    echo ""
    
    # Export VENV_ROOT for calling scripts
    export VENV_ROOT
    return 0
}

# Run verification if script is executed directly (not sourced)
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    verify_venv
    exit $?
else
    # If sourced, just run the function
    verify_venv
fi
