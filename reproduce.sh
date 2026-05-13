#!/bin/bash
# reproduce.sh - One-command reproducibility for TongueVision OS Research

echo "=== Initializing TongueVision Reproducibility Suite ==="

# 1. Check for Model Weights
if [ ! -f "TongueVision_Diabetes_Final.pth" ]; then
    echo "Error: TongueVision_Diabetes_Final.pth not found!"
    exit 1
fi

# 2. Check for Dependencies
echo "Checking Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# 3. Check for Perf Permissions
PERF_LEVEL=$(cat /proc/sys/kernel/perf_event_paranoid)
if [ "$PERF_LEVEL" -gt 1 ]; then
    echo "Warning: perf_event_paranoid is set to $PERF_LEVEL."
    echo "Please run: sudo sysctl -w kernel.perf_event_paranoid=-1"
    exit 1
fi

# 4. Run the Benchmarking Pipeline
echo "Starting Benchmarking (This will take ~1 hour due to cooldowns)..."
bash bench.sh

# 5. Parse and Generate Tables
echo "Generating Final Tables..."
python3 parse_results.py

echo "=== Reproduction Complete. See benchmark_results.log for raw data. ==="