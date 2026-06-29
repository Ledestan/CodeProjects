import sys

sys.dont_write_bytecode = True

from flask_wtf import FlaskForm
from wtforms import (DateTimeField, IntegerField, SelectField, StringField,
                     SubmitField, TextAreaField)
from wtforms.validators import DataRequired, NumberRange


class RegistrationForm(FlaskForm):
    title = StringField("标题", validators=[DataRequired()])
    type = SelectField(
        "类型",
        choices=[("registration", "报名"), ("checkin", "签到")],
        validators=[DataRequired()],
    )
    max_participants = IntegerField(
        "人数上限", validators=[NumberRange(min=1)], default=50
    )
    deadline = DateTimeField(
        "截止时间", format="%Y-%m-%dT%H:%M", validators=[DataRequired()]
    )
    description = TextAreaField("描述")
    event_id = IntegerField(
        "关联活动ID", validators=[DataRequired()]
    )  # 简易版：手动输入活动ID
    submit = SubmitField("创建")
