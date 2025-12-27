echo "Time is $(date +"%H:%M %d-%m-%y") ($(date +%s))"

export GEMINI_MODE=submit
python src-new/main.py --tasks_file closed_models_tasks_files/tasks_gemini-2.5-flash.json