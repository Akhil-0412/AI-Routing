$ServerPort = 8000
$OutputFile = "test_results.txt"
$EnvFile = ".env"

function Start-Server {
    Write-Host "Starting server..."
    Start-Process -FilePath "uv" -ArgumentList "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$ServerPort", "--env-file", "$EnvFile" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

function Stop-Server {
    Write-Host "Stopping server..."
    Stop-Process -Name python -Force -ErrorAction SilentlyContinue
    Stop-Process -Name uvicorn -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Run-LoadGenerator {
    param([string]$LoadArgs)
    Write-Output "`n=======================================================" | Out-File -FilePath $OutputFile -Append
    Write-Output "RUNNING: load_generator.py $LoadArgs" | Out-File -FilePath $OutputFile -Append
    Write-Output "=======================================================" | Out-File -FilePath $OutputFile -Append
    
    $Output = Invoke-Expression "uv run python load_generator.py $LoadArgs 2>&1"
    $Output | Out-File -FilePath $OutputFile -Append
}

Clear-Content -Path $OutputFile -ErrorAction SilentlyContinue

# Ensure clean slate
Stop-Server

Start-Server
Run-LoadGenerator -LoadArgs ""
Stop-Server

Start-Server
Run-LoadGenerator -LoadArgs "--tight-sla"
Stop-Server

Start-Server
Run-LoadGenerator -LoadArgs "--failure-demo"
Stop-Server

Start-Server
Run-LoadGenerator -LoadArgs "--latency-breach"
Stop-Server

Start-Server
Run-LoadGenerator -LoadArgs "--stress"
Stop-Server

Write-Host "All done! Results saved to $OutputFile"
