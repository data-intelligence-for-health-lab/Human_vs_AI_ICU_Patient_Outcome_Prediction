#!/bin/zsh
# ENV_NAME="myenv"
# if [ -z "$VIRTUAL_ENV" ]; then
#     echo "activating $ENV_NAME..."
#     pyenv activate $ENV_NAME
# else
#     echo "virtual environment already active: $VIRTUAL_ENV"
# fi
# pip freeze > server_requirements.txt
rsync -avz  --exclude-from='./dih_lab_server/exclude.txt' LOCALPATH lu.kuang@10.44.113.28:/home/thesis/GithubFolder

#--delete
#from to 



