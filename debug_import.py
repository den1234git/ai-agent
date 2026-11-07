import traceback, sys
print("=== app.py import test ===")
try:
    import app
    print("IMPORT OK - app.py は正常に読み込めました")
except Exception:
    print("IMPORT FAILED - 以下が原因です:")
    traceback.print_exc()
    sys.exit(1)
