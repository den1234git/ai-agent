taskkill /f /im python.exe 2>$null
Start-Sleep 1
$python = "C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe"
Start-Process $python -ArgumentList "-m uvicorn app:app --host 0.0.0.0 --port 8000" -WorkingDirectory "F:\ai_agent\server" -WindowStyle Hidden
Start-Sleep 3
Write-Host "Server restarted. Testing..."
Start-Sleep 3
try { Invoke-WebRequest -Uri "http://localhost:8000/status" -UseBasicParsing | Out-Null; Write-Host "OK: server is up" } catch { Write-Host "NG: server not responding" }
