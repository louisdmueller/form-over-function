echo "Time is $(date +"%H:%M %d-%m-%y") ($(date +%s))"

export GEMINI_MODE=retrieve
python src/main.py --tasks_file closed_models_tasks_files/tasks_gemini-2.5-flash.json