echo "Time is $(date +"%H:%M %d-%m-%y") ($(date +%s))"

export ANTHROPIC_MODE=retrieve
python src-new/main.py --tasks_file closed_models_tasks_files/tasks_Claude_Haiku_4.5.json