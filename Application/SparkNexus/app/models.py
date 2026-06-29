import sys

sys.dont_write_bytecode = True

from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from . import db


# 角色常量
class Role:
    ADMIN = "admin"
    TEACHER = "teacher"
    MONITOR = "monitor"  # 班委
    LEADER = "leader"  # 临时负责人
    STUDENT = "student"

    # 等级映射
    LEVELS = {
        ADMIN: 100,
        TEACHER: 80,
        MONITOR: 60,
        LEADER: 40,
        STUDENT: 20,
    }

    @classmethod
    def get_level(cls, role_name):
        return cls.LEVELS.get(role_name, 0)


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=Role.STUDENT)
    class_id = db.Column(db.Integer, default=0)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def change_password(self, old_password, new_password):
        if self.check_password(old_password):
            self.set_password(new_password)
            return True
        return False

    def is_admin(self):
        return self.role == Role.ADMIN

    def __repr__(self):
        return f"<User {self.username} {self.role}>"

    @property
    def role_level(self):
        return Role.get_level(self.role)

    def has_permission(self, required_role):
        """检查当前用户是否满足至少 required_role 的权限等级"""
        return self.role_level >= Role.get_level(required_role)


class Event(db.Model):
    __tablename__ = "events"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # course / activity / personal
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    description = db.Column(db.Text)
    visibility = db.Column(db.String(20), default="public")  # public / class / private
    creator_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    class_id = db.Column(db.Integer, default=0)

    creator = db.relationship("User", backref="events")


class Registration(db.Model):
    __tablename__ = "registrations"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # registration / checkin
    max_participants = db.Column(db.Integer)
    deadline = db.Column(db.DateTime)
    creator_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"))
    class_id = db.Column(db.Integer, default=0)

    creator = db.relationship("User", backref="registrations")


class RegistrationRecord(db.Model):
    __tablename__ = "registration_records"
    id = db.Column(db.Integer, primary_key=True)
    registration_id = db.Column(db.Integer, db.ForeignKey("registrations.id"))
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    status = db.Column(db.String(20), default="signed_up")  # signed_up / checked_in

    user = db.relationship("User", backref="records")
    registration = db.relationship("Registration", backref="records")


class Announcement(db.Model):
    __tablename__ = "announcements"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text)
    type = db.Column(db.String(20), nullable=False)  # daily / homework / activity
    sender_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    class_id = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

    sender = db.relationship("User", backref="announcements")
