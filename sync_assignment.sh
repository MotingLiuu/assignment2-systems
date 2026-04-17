#!/bin/bash
rsync -avzP \
  --include="cs336_systems/***" \
  --include="experiments/***" \
  --exclude="*" \
  ~/workspace/CS336/assignment2-systems/ mutyuu@fairy:~/workspace/CS336/assignment2-systems/
