import sys

sys.dont_write_bytecode = True

from flask import Flask, render_template
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

from .config import Config

db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()
csrf = CSRFProtect()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = "请先登录。"

    # 显式导入 models 模块，确保所有模型类被 SQLAlchemy 识别
    from . import models  # 注册 Event、Registration 等模型
    # 注册蓝图
    from .auth import auth_bp

    app.register_blueprint(auth_bp)

    from .schedule import schedule_bp

    app.register_blueprint(schedule_bp)

    from .registration import registration_bp

    app.register_blueprint(registration_bp)

    from .notification import notification_bp

    app.register_blueprint(notification_bp)

    # 用户加载器（只需要 User 类）
    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # 主页路由（只保留一个）
    @app.route("/")
    def home():
        return render_template("index.html")

    return app
