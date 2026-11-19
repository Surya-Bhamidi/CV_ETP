"""Start or stop the local comparison server. See backend/api.py for its routes."""

from backend.api import app, main

if __name__ == "__main__":
    main()
