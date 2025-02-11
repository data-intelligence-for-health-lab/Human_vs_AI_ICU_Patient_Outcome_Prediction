#!/bin/zsh
# ENV_NAME="myenv"
# if [ -z "$VIRTUAL_ENV" ]; then
#     echo "activating $ENV_NAME..."
#     pyenv activate $ENV_NAME
# else
#     echo "virtual environment already active: $VIRTUAL_ENV"
# fi
# pip freeze > server_requirements.txt
rsync -avz --delete --exclude-from='./dih_lab_server/exclude.txt' LOCALPATH YOU@10.44.113.28:/home/corin....






