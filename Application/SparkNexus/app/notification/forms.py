import sys

sys.dont_write_bytecode = True

from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length


class AnnouncementForm(FlaskForm):
    title = StringField("标题", validators=[DataRequired(), Length(max=100)])
    content = TextAreaField("内容", validators=[DataRequired()])
    type = SelectField("类型", coerce=str)
    submit = SubmitField("发布")

    def set_type_choices(self, role):
        """根据角色动态设置可发布的通知类型"""
        if role == "leader":
            self.type.choices = [("activity", "活动通知")]
        else:
            self.type.choices = [
                ("daily", "日常通知"),
                ("homework", "作业通知"),
                ("activity", "活动通知"),
            ]
