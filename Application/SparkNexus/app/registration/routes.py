import sys

sys.dont_write_bytecode = True

from datetime import datetime

from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from ..decorators import require_roles
from ..models import Event, Registration, RegistrationRecord, db
from . import registration_bp
from .forms import RegistrationForm


@registration_bp.route("/")
@login_required
def list():
    registrations = Registration.query.order_by(Registration.deadline).all()
    return render_template("registration/list.html", registrations=registrations)


@registration_bp.route("/create", methods=["GET", "POST"])
@login_required
@require_roles("leader")
def create():
    form = RegistrationForm()
    if form.validate_on_submit():
        # 检查活动是否存在
        event = Event.query.get(form.event_id.data)
        if not event:
            flash("关联活动不存在。")
            return render_template("registration/create.html", form=form)
        reg = Registration(
            title=form.title.data,
            type=form.type.data,
            max_participants=form.max_participants.data,
            deadline=form.deadline.data,
            creator_id=current_user.id,
            event_id=event.id,
            class_id=current_user.class_id,
        )
        db.session.add(reg)
        db.session.commit()
        flash("报名/签到创建成功。")
        return redirect(url_for("registration.list"))
    return render_template("registration/create.html", form=form)


@registration_bp.route("/<int:id>")
@login_required
def detail(id):
    reg = Registration.query.get_or_404(id)
    # 学生操作：报名/签到
    if request.method == "POST":
        return handle_signup(reg)  # 函数在下方
    records = RegistrationRecord.query.filter_by(registration_id=id).all()
    return render_template("registration/detail.html", reg=reg, records=records)


def handle_signup(reg):
    if current_user.role != "student":
        flash("只有学生可以参与报名/签到。")
        return redirect(url_for("registration.detail", id=reg.id))
    # 检查截止时间
    if reg.deadline and datetime.utcnow() > reg.deadline:
        flash("已超过截止时间。")
        return redirect(url_for("registration.detail", id=reg.id))
    # 检查人数上限
    count = RegistrationRecord.query.filter_by(registration_id=reg.id).count()
    if reg.max_participants and count >= reg.max_participants:
        flash("名额已满。")
        return redirect(url_for("registration.detail", id=reg.id))
    # 防止重复报名
    existing = RegistrationRecord.query.filter_by(
        registration_id=reg.id, user_id=current_user.id
    ).first()
    if existing:
        flash("您已经报过名/签过到。")
        return redirect(url_for("registration.detail", id=reg.id))
    record = RegistrationRecord(
        registration_id=reg.id,
        user_id=current_user.id,
        status="signed_up" if reg.type == "registration" else "checked_in",
    )
    db.session.add(record)
    db.session.commit()
    flash("操作成功！")
    return redirect(url_for("registration.detail", id=reg.id))


@registration_bp.route("/<int:id>/signup", methods=["POST"])
@login_required
def signup(id):
    """学生点击报名/签到按钮"""
    reg = Registration.query.get_or_404(id)
    return handle_signup(reg)
