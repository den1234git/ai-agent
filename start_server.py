import sys
import os

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SERVER_DIR)
sys.path.insert(0, SERVER_DIR)

# Windows で子プロセスが新しい CMD ウィンドウを開かないようにする
if sys.platform == "win32":
    import subprocess
    _orig_init = subprocess.Popen.__init__

    def _no_window_init(self, *args, **kwargs):
        cf = kwargs.get("creationflags", 0)
        kwargs["creationflags"] = cf | 0x08000000  # CREATE_NO_WINDOW
        _orig_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _no_window_init

import uvicorn

uvicorn.run("app:app", host="127.0.0.1", port=8765)
