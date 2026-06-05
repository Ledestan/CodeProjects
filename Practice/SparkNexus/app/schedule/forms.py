import sys

sys.dont_write_bytecode = True

from flask_wtf import FlaskForm
from wtforms import DateTimeField, SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired


class EventForm(FlaskForm):
    title = StringField("标题", validators=[DataRequired()])
    type = SelectField(
        "类型",
        choices=[("course", "课程"), ("activity", "活动"), ("personal", "个人日程")],
        validators=[DataRequired()],
    )
    start_time = DateTimeField(
        "开始时间", format="%Y-%m-%dT%H:%M", validators=[DataRequired()]
    )
    end_time = DateTimeField(
        "结束时间", format="%Y-%m-%dT%H:%M", validators=[DataRequired()]
    )
    description = TextAreaField("描述")
    visibility = SelectField(
        "可见范围",
        choices=[("public", "公开"), ("class", "班级"), ("private", "仅自己")],
    )
    submit = SubmitField("保存")
