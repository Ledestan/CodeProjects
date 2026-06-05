import sys

sys.dont_write_bytecode = True

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from ..decorators import require_roles
from ..models import Announcement, db
from . import notification_bp
from .forms import AnnouncementForm


@notification_bp.route("/")
@login_required
def list_all():
    page = request.args.get("page", 1, type=int)
    type_filter = request.args.get("type")
    query = Announcement.query
    if type_filter and type_filter in ["daily", "homework", "activity"]:
        query = query.filter(Announcement.type == type_filter)
    announcements = query.order_by(Announcement.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template(
        "notification/list.html", announcements=announcements, type_filter=type_filter
    )


@notification_bp.route("/create", methods=["GET", "POST"])
@login_required
@require_roles("leader")
def create():
    form = AnnouncementForm()
    form.set_type_choices(current_user.role)  # 关键：注入动态选择
    if form.validate_on_submit():
        announcement = Announcement(
            title=form.title.data,
            content=form.content.data,
            type=form.type.data,
            sender_id=current_user.id,
            class_id=current_user.class_id,
        )
        db.session.add(announcement)
        db.session.commit()
        flash("通知发布成功。")
        return redirect(url_for("notification.list_all"))
    return render_template("notification/create.html", form=form)


@notification_bp.route("/<int:id>")
@login_required
def detail(id):
    announcement = Announcement.query.get_or_404(id)
    return render_template("notification/detail.html", announcement=announcement)
