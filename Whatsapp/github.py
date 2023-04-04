from selenium import webdriver
from time import sleep
import os
from password import username, password

class GitHub:
    def __init__(self, username, pw, pname):
        self.driver = webdriver.Chrome('C://chromedriver.exe')
        
        self.username = username
        self.driver.get("https://github.com/login")
        sleep(2)
        sleep(2)
        self.driver.find_element_by_xpath("//*[@id=\"login_field\"]")\
            .send_keys(username)
        self.driver.find_element_by_xpath("//*[@id=\"password\"]")\
            .send_keys(pw)
        self.driver.find_element_by_xpath('//*[@id="login"]/div[4]/form/input[14]')\
            .click()
        sleep(4)
        self.driver.find_element_by_xpath("//*[@id=\"repos-container\"]/h2/a")\
            .click()
        sleep(2)
        self.driver.find_element_by_xpath("//*[@id=\"repository_name\"]")\
            .send_keys(pname)
        sleep(2)
        self.driver.find_element_by_xpath("//*[@id=\"new_repository\"]/div[4]/button")\
            .click()
        sleep(2)
        element = self.driver.find_element_by_xpath("//*[@id=\"empty-setup-push-repo-echo\"]/span[1]")
        git = element.text
        self.git_push(git)
        self.driver.refresh()

    def git_push(self,push_url):
        os.chdir(path) 
        os.system("git init")
        os.system("git status")
        os.system("git add .")
        os.system("git status")
        os.system('git commit -m "v1" .')
        os.system("git status")
        os.system(push_url)
        os.system("git push -u origin master")

        
        

email = username
password = password
repo = input("Enter repository name:- ")   
path = input("Enter path to directory:- ")

print("------------START------------")
my_bot = GitHub(email, password, repo)



