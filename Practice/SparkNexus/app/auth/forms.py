import sys

sys.dont_write_bytecode = True

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    HiddenField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
)
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional


class LoginForm(FlaskForm):
    username = StringField("用户名", validators=[DataRequired()])
    password = PasswordField("密码", validators=[DataRequired()])
    remember = BooleanField("记住我")
    submit = SubmitField("登录")


class RegistrationForm(FlaskForm):
    username = StringField("用户名", validators=[DataRequired(), Length(min=3, max=64)])
    password = PasswordField("密码", validators=[DataRequired(), Length(min=6)])
    confirm = PasswordField(
        "确认密码", validators=[DataRequired(), EqualTo("password")]
    )
    submit = SubmitField("注册")


class ChangePasswordForm(FlaskForm):
    old_password = PasswordField("当前密码", validators=[DataRequired()])
    new_password = PasswordField("新密码", validators=[DataRequired(), Length(min=6)])
    confirm = PasswordField(
        "确认新密码", validators=[DataRequired(), EqualTo("new_password")]
    )
    submit = SubmitField("修改密码")


class EditUserRoleForm(FlaskForm):
    role = SelectField(
        "角色",
        choices=[
            ("student", "学生"),
            ("teacher", "教师"),
            ("monitor", "班委"),
            ("leader", "临时负责人"),
            ("admin", "管理员"),
        ],
        validators=[DataRequired()],
    )
    submit = SubmitField("更新")


class AdminResetPasswordForm(FlaskForm):
    new_password = PasswordField("新密码", validators=[DataRequired(), Length(min=6)])
    confirm = PasswordField(
        "确认新密码", validators=[DataRequired(), EqualTo("new_password")]
    )
    submit = SubmitField("重置密码")
