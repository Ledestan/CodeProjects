import sys

sys.dont_write_bytecode = True

from flask import Blueprint

notification_bp = Blueprint("notification", __name__, url_prefix="/notifications")
from . import routes
