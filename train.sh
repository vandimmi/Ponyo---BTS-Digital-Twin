#!/bin/bash
# Kaggle/Local training wrapper script

# Set seed for reproducibility
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unique output dir based on timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=== Training Run Logs ===" > "$OUTPUT_DIR/run_log.txt"
echo "Date: $(date)" >> "$OUTPUT_DIR/run_log.txt"
echo "Seed: 42" >> "$OUTPUT_DIR/run_log.txt"

# Log environment
echo "=== Environment Info ===" >> "$OUTPUT_DIR/run_log.txt"
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')" >> "$OUTPUT_DIR/run_log.txt" 2>&1
echo "=== PIP FREEZE ===" >> "$OUTPUT_DIR/run_log.txt"
pip freeze >> "$OUTPUT_DIR/pip_freeze.txt"

echo "Starting training..."
# Run 3DGS training, piping output to log
# Using -m output to use our timestamped directory
python gaussian-splatting/train.py -s data/HCM0421 -m "$OUTPUT_DIR" --iterations 100 2>&1 | tee -a "$OUTPUT_DIR/run_log.txt"

# If it runs successfully, auto-zip
echo "Packaging run logs..."
if [ -d "/kaggle/working" ]; then
    ZIP_PATH="/kaggle/working/run_${TIMESTAMP}.zip"
else
    ZIP_PATH="${OUTPUT_DIR}_package.zip"
fi

zip -r "$ZIP_PATH" "$OUTPUT_DIR"
echo "Finished! Logs and outputs saved to $ZIP_PATH"
