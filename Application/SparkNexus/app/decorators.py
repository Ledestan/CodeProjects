import sys

sys.dont_write_bytecode = True

from functools import wraps

from flask import abort
from flask_login import current_user


def require_roles(*roles):
    """允许拥有任一角色（或更高等级）的用户访问"""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(403)
            # 用户只要满足其中某一个角色的等级即可
            if not any(current_user.has_permission(role) for role in roles):
                abort(403)
            return f(*args, **kwargs)

        return decorated_function

    return decorator
