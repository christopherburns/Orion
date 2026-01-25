#!/bin/bash

# Build script for Orion
# Usage: ./build.sh [debug|release]
# Default: release

CONFIGURATION="${1:-release}"

if [[ "$CONFIGURATION" != "debug" && "$CONFIGURATION" != "release" ]]; then
    echo "Error: Configuration must be 'debug' or 'release'"
    echo "Usage: $0 [debug|release]"
    exit 1
fi

echo "Building Orion ($CONFIGURATION)..."

xcodebuild build \
    -scheme Orion \
    -configuration "$CONFIGURATION" \
    -destination 'platform=macOS' \
    CONFIGURATION_BUILD_DIR="$PWD/.build/$CONFIGURATION"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build succeeded!"
    echo "  Binary: .build/$CONFIGURATION/orion"
    echo "  MetalLib: .build/$CONFIGURATION/mlx-swift_Cmlx.bundle"
    echo ""
    echo "Run with: .build/$CONFIGURATION/orion"
else
    echo ""
    echo "✗ Build failed"
    exit 1
fi
