import sys

sys.dont_write_bytecode = True

from flask import Blueprint

registration_bp = Blueprint("registration", __name__, url_prefix="/registrations")
from . import routes
