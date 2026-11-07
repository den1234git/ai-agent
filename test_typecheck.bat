@echo off
cd /d %~dp0
echo _typecheck_code() の動作確認テストを実行します（Ollamaは使いません）...
echo.
python test_typecheck.py
echo.
pause
