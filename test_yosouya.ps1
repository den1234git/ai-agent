$body = '{"question":"2026 US midterm election: will Democrats win the House?","deadline":"2026-11-04"}'
$response = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/yosouya" -ContentType "application/json" -Body $body -TimeoutSec 180
$response | ConvertTo-Json -Depth 5
