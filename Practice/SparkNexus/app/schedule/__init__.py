import sys

sys.dont_write_bytecode = True

from flask import Blueprint

schedule_bp = Blueprint("schedule", __name__, url_prefix="/schedule")
from . import routes
