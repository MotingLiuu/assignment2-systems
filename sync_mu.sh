#!/bin/bash

# 使用 $HOME 获取本地绝对路径
LOCAL_DIR="$HOME/workspace/CS336/assignment2-systems"
# 远程端严格使用绝对路径 /home/mutyuu
REMOTE_DIR="mutyuu@fairy:/home/mutyuu/workspace/CS336/assignment2-systems"

echo "Starting synchronization for cs336_systems..."
mutagen sync create --name=sync-systems \
  "$LOCAL_DIR/cs336_systems/" \
  "$REMOTE_DIR/cs336_systems/"

sleep 20

echo "Starting synchronization for experiments..."
mutagen sync create --name=sync-experiments \
  "$LOCAL_DIR/experiments/" \
  "$REMOTE_DIR/experiments/"

echo "Sync sessions created! Check status with: mutagen sync list"
