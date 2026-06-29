@echo off
echo 正在停止并清除旧数据...
docker-compose down -v
echo 正在启动最新数据库...
docker-compose up -d
echo 数据库已就绪！
pause