#!/bin/bash

for file in configs/experiment/MTI_TargetNet_Optimized.yaml configs/experiment/deepTargetPro_TargetNet_Optimized.yaml; do
    if [ -f "$file" ]; then
        sed -i 's|name: targetnet_optimized|name: glm-5.1.2|g' "$file"
        echo "✓ Updated: $file"
    else
        echo "⚠ Not found: $file"
    done
