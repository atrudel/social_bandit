if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo -e "\033[31mCOMMIT all your changes before you run the training script.\033[0m"
  exit 1
fi
commit_hash=$(git rev-parse --short=7 HEAD)
export COMMIT_HASH=$commit_hash