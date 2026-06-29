import sys

sys.dont_write_bytecode = True

from datetime import datetime

from flask import (abort, flash, jsonify, redirect, render_template, request,
                   url_for)
from flask_login import current_user, login_required

from ..decorators import require_roles
from ..models import Event, db
from . import schedule_bp
from .forms import EventForm


@schedule_bp.route("/")
@login_required
def view():
    """课表查询 - 返回所有公开/班级事件 + 自己的私有事件"""
    events = (
        Event.query.filter(
            (Event.visibility.in_(["public", "class"]))
            | ((Event.visibility == "private") & (Event.creator_id == current_user.id))
        )
        .order_by(Event.start_time)
        .all()
    )
    return render_template("schedule/view.html", events=events)


@schedule_bp.route("/manage")
@login_required
@require_roles("admin", "teacher")
def manage():
    """课表调整页面"""
    events = Event.query.filter_by(type="course").order_by(Event.start_time).all()
    return render_template("schedule/manage.html", events=events)


@schedule_bp.route("/api/events", methods=["POST"])
@login_required
@require_roles("admin", "teacher")
def create_course_event():
    """通过 AJAX 创建/更新课表（用于课表调整）"""
    data = request.get_json()
    event = Event(
        title=data["title"],
        type="course",
        start_time=datetime.fromisoformat(data["start"]),
        end_time=datetime.fromisoformat(data["end"]),
        description=data.get("description", ""),
        visibility="class",
        creator_id=current_user.id,
        class_id=current_user.class_id,
    )
    db.session.add(event)
    db.session.commit()
    return jsonify({"id": event.id}), 201


@schedule_bp.route("/api/events/<int:id>", methods=["PUT", "DELETE"])
@login_required
@require_roles("admin", "teacher")
def modify_course_event(id):
    event = Event.query.get_or_404(id)
    if event.type != "course":
        abort(400)
    if request.method == "DELETE":
        db.session.delete(event)
        db.session.commit()
        return jsonify({"message": "deleted"}), 200
    # PUT update
    data = request.get_json()
    event.title = data["title"]
    event.start_time = datetime.fromisoformat(data["start"])
    event.end_time = datetime.fromisoformat(data["end"])
    event.description = data.get("description", "")
    db.session.commit()
    return jsonify({"id": event.id}), 200


@schedule_bp.route("/add", methods=["GET", "POST"])
@login_required
@require_roles("student")
def add_personal():
    form = EventForm()
    if current_user.role == "student":
        # 学生只能添加个人日程
        form.type.choices = [("personal", "个人日程")]
        form.visibility.choices = [("private", "仅自己")]
    if form.validate_on_submit():
        event = Event(
            title=form.title.data,
            type=form.type.data,
            start_time=form.start_time.data,
            end_time=form.end_time.data,
            description=form.description.data,
            visibility=form.visibility.data,
            creator_id=current_user.id,
            class_id=current_user.class_id,
        )
        db.session.add(event)
        db.session.commit()
        flash("日程添加成功。")
        return redirect(url_for("schedule.view"))
    return render_template("schedule/add.html", form=form)


@schedule_bp.route("/create-event", methods=["GET", "POST"])
@login_required
@require_roles("leader")
def create_activity():
    """发布活动（activity 类型）"""
    form = EventForm()
    # 强制类型为活动
    form.type.data = "activity"
    form.type.render_kw = {"disabled": "disabled"}
    if form.validate_on_submit():
        event = Event(
            title=form.title.data,
            type="activity",
            start_time=form.start_time.data,
            end_time=form.end_time.data,
            description=form.description.data,
            visibility=form.visibility.data,
            creator_id=current_user.id,
            class_id=current_user.class_id,
        )
        db.session.add(event)
        db.session.commit()
        flash("活动发布成功。")
        return redirect(url_for("schedule.view"))
    return render_template("schedule/create_event.html", form=form)
