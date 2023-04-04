import os

os.chdir("E:\Bot") 

os.system("git init")
os.system("git status")
os.system("git add .")
os.system("git status")
os.system('git commit -m "v1" .')
os.system("git status")
os.system("git remote add origin https://github.com/omkarbhope/Bot.git")
os.system("git push -u origin master")