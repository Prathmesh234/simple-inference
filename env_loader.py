import os
from pathlib import Path

def load_env():
    # Find .env by checking current directory up to 5 levels above
    current = Path(__file__).resolve().parent
    for _ in range(5):
        env_path = current / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip()
                            # Strip outer quotes if they exist
                            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                                val = val[1:-1]
                            # Don't clobber variables already set in the shell
                            # environment — a real `VAR=... command` prefix should
                            # win over .env (standard dotenv override=False).
                            os.environ.setdefault(key, val)
            except Exception:
                pass
            break
        current = current.parent

# Load environment variables on module import
load_env()
