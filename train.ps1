$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "output/run_$timestamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$logFile = "$outDir/run_log.txt"
"=== Training Run Logs ===" | Out-File -FilePath $logFile
"Date: $(Get-Date)" | Out-File -FilePath $logFile -Append
"Seed: 42" | Out-File -FilePath $logFile -Append

"=== Environment Info ===" | Out-File -FilePath $logFile -Append
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')" >> $logFile 2>&1

"=== PIP FREEZE ===" | Out-File -FilePath $logFile -Append
pip freeze > "$outDir/pip_freeze.txt"

Write-Host "Starting training..."
# This will crash because of diff-gaussian-rasterization, but we capture the crash log!
python gaussian-splatting/train.py -s data/HCM0421 -m $outDir --iterations 100 >> $logFile 2>&1

Write-Host "Packaging run logs..."
Compress-Archive -Path "$outDir" -DestinationPath "${outDir}_package.zip" -Force
Write-Host "Finished! Logs and outputs saved to ${outDir}_package.zip"
