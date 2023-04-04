from selenium import webdriver
from time import sleep
import os
from password import username, password
from selenium.webdriver.support.select import Select
import threading
import random

class Xenia:
    def __init__(self, username, pw,n):
        n = str(n)
        c = webdriver.ChromeOptions()
        c.add_argument("--incognito")
        self.driver = webdriver.Chrome('chromedriver.exe',options=c)
        self.driver.implicitly_wait(0.5)
        self.username = username
        self.driver.get("https://xeniamcq.co.in")
        sleep(2)
        self.driver.find_element_by_xpath("//*[@id='formBasicEmail']")\
            .send_keys(username)
        self.driver.find_element_by_xpath("//*[@id='formBasicPassword']")\
            .send_keys(pw)
        # self.driver.find_element_by_class_name('MuiSelect-root MuiSelect-select MuiSelect-selectMenu MuiInputBase-input MuiInput-input').click()
        self.driver.find_element_by_xpath("//*[@id='root']/div/div/div/div/div/div[2]/div/form/div[3]/div")\
            .click()
        self.driver.find_element_by_xpath(f"//*[@id='menu-']/div[3]/ul/li[{n}]")\
            .click()
        sleep(2)
        self.driver.find_element_by_xpath("//*[@id='root']/div/div/div/div/div/div[2]/div/form/button")\
            .click()
        sleep(2)


n = random.randint(2,5)

email = username
password = password[n-2]
print("------------START------------")
xenai1 = Xenia(email, password,n)

