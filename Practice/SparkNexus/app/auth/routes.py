import sys

sys.dont_write_bytecode = True

from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from .. import db
from ..decorators import require_roles
from ..models import Role, User
from . import auth_bp
from .forms import (
    AdminResetPasswordForm,
    ChangePasswordForm,
    EditUserRoleForm,
    LoginForm,
    RegistrationForm,
)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))  # 改为 home
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            flash("登录成功。")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("home"))
        flash("用户名或密码错误。")
    return render_template("auth/login.html", form=form)


@auth_bp.route("/register", methods=["GET", "POST"])
@login_required  # 仅登录后可访问
@require_roles("admin")  # 仅管理员可添加用户
def register():
    # 如果管理员已登录，直接显示注册表单（无需跳转）
    form = RegistrationForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash("用户名已存在。")
            return render_template("auth/register.html", form=form)

        user = User(
            username=form.username.data,
            role=Role.STUDENT,  # 强制学生，管理员可在后台改角色
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("用户创建成功。")
        return redirect(url_for("auth.manage_users"))  # 创建后返回用户管理页
    return render_template("auth/register.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("已退出登录。")
    return redirect(url_for("home"))


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if current_user.change_password(form.old_password.data, form.new_password.data):
            db.session.commit()
            flash("密码修改成功。")
            return redirect(url_for("auth.profile"))
        else:
            flash("当前密码错误。")
    return render_template("auth/profile.html", form=form)


@auth_bp.route("/admin/users")
@login_required
@require_roles("admin")  # 统一用字符串
def manage_users():
    users = User.query.order_by(User.username).all()
    return render_template("auth/admin_users.html", users=users)


@auth_bp.route("/admin/users/<int:user_id>/edit", methods=["GET", "POST"])
@login_required
@require_roles("admin")
def edit_user_role(user_id):
    user = User.query.get_or_404(user_id)
    if user == current_user:
        flash("不能修改自己的角色。")
        return redirect(url_for("auth.manage_users"))
    form = EditUserRoleForm()
    if form.validate_on_submit():
        user.role = form.role.data
        db.session.commit()
        flash(f"用户 {user.username} 的角色已更新为 {user.role}。")
        return redirect(url_for("auth.manage_users"))
    elif request.method == "GET":
        form.role.data = user.role
    return render_template("auth/edit_role.html", form=form, user=user)


@auth_bp.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@login_required
@require_roles("admin")
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user == current_user:
        flash("不能删除自己的账号。")
    else:
        db.session.delete(user)
        db.session.commit()
        flash(f"用户 {user.username} 已删除。")
    return redirect(url_for("auth.manage_users"))


@auth_bp.route("/admin/users/<int:user_id>/reset-password", methods=["GET", "POST"])
@login_required
@require_roles("admin")
def reset_user_password(user_id):
    user = User.query.get_or_404(user_id)
    form = AdminResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.new_password.data)
        db.session.commit()
        flash(f"用户 {user.username} 的密码已重置。")
        return redirect(url_for("auth.manage_users"))
    return render_template("auth/reset_password.html", form=form, user=user)
