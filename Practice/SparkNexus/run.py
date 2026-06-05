import sys

sys.dont_write_bytecode = True

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
