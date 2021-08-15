import time
import re
import csv
import json
import random
import pandas as pd
from bs4.element import ResultSet, TemplateString
import requests
from bs4 import BeautifulSoup
from lxml import html

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import utils

class ImageTypeError(Exception):
    """画像のtypeが選択されていないことを知らせる例外クラス"""
    pass

def transition_page(driver, current_url):
    # ページ遷移ボタンがnav tagなのでnav要素をget
    elements = driver.find_elements_by_xpath("//nav")
    # 次に行くボタンのurlをもらって遷移する
    a_tag = elements[-1].find_element_by_tag_name("a")
    url = a_tag.get_attribute("href")

    if url == current_url:
        return None
    else:
        return url

def get_image(save_dir, url, image_count):
    file_name = save_dir + str(image_count) + ".jpg"

    response = requests.get(url)
    image = response.content

    with open(file_name, "wb") as aaa:
        aaa.write(image)

def tag2binary(image_tags, search_tags):
    binary_list = [1 if i in image_tags else 0 for i in search_tags]
    return binary_list

def main():
    # Seleniumをあらゆる環境で起動させるChromeオプション
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")
    options.add_argument('--start-maximized')
    options.add_argument('--headless')

    # ブラウザの起動
    driver = webdriver.Chrome(executable_path='./chromedriver',chrome_options=options)

    driver.get("https://nijie.info/login.php")
    print('Email : ', end='')
    login_user = input()
    print('¥n')
    print('Password : ', end='')
    login_pass = input()
    print('¥n')

    # 各タグごとの持ってくる画像の上限枚数
    image_limit = 3

    time.sleep(5)

    # ログインボタンをクリックしてからemailとpasswordを入力する
    element = driver.find_element_by_class_name("ok").click()
    element = driver.find_element_by_id("slide_login_button").click()
    element = driver.find_element_by_xpath("//input[@name='email']")
    element.send_keys(login_user)
    time.sleep(3)
    element = driver.find_element_by_xpath("//input[@name='password']")
    element.send_keys(login_pass)
    time.sleep(1)
    element.send_keys(Keys.ENTER)

    time.sleep(5)

    with open('access_tag.csv', 'r') as f:
        reader = csv.reader(f)
        search_tags = [row for row in reader][0]

    utils.mkdir('./img', delete=True)
    save_path = './img/'
    total_img_counter = 0
    columns = ['path', 'label']
    # 画像のlabelとなるcsvファイルを作成
    images_info = []

    for search_tag in search_tags:
        print('---------------')
        print('start search tag: {}'.format(search_tag))
        print('---------------')
        image_limit_flag = False
        img_counter = 0
        url = "http://nijie.info/search.php?type=&word="+search_tag+"&p=1&mode=&illust_type=1&sort=1&pd=&con="
        driver.get(url)
        
        while True:
            url = driver.current_url
            # 画像を保存
            res = requests.get(url)
            soup = BeautifulSoup(res.content, 'html.parser')
            time.sleep(5)
            urls = []
            divs = soup.find_all('div', class_='picture')
            for div in divs:
                urls.append(div.find('a'))
            for url in urls:
                time.sleep(random.randrange(5,15))
                href  = url.get("href")
                img_url = requests.get('http://nijie.info'+href)
                img_soup = BeautifulSoup(img_url.content, 'html.parser')
                img = img_soup.find_all('img', class_='mozamoza ngtag')
                img = re.sub('__.+?/', '', img[0].get('src'))
                response = requests.get('http:' + img)
                tags = img_soup.find_all('span', class_='tag_name')
                tag_length = int(len(tags)//2)
                img_label = [search_tag]
                for tag in tags[:tag_length]:
                    tag_name = tag.find('a').get_text()
                    if tag_name in search_tags:
                        img_label.append(tag_name)
                # 重複するタグの削除
                img_label = list(set(img_label))
                
                if img_counter >= image_limit:
                    image_limit_flag = True

                if image_limit_flag:
                    break
                else:
                    with open(save_path + "{}.jpg".format(total_img_counter), "wb") as f:
                        print('save figure!')
                        images_info.append(['./img/' + str(total_img_counter) + '.jpg'] + [tag2binary(img_label, search_tags)])
                        img_counter += 1
                        total_img_counter += 1
                        f.write(response.content)

            if image_limit_flag:
                break

            # page遷移
            if driver.find_element_by_class_name('right'):
                element = driver.find_element_by_class_name('right').click()
            else:
                break
        # elements = driver.find_elements_by_class_name("clearfix")
        # for element in elements:
        #     img_tags = element.find_elements_by_tag_name('img')
        #     print(img_tags)
        #     dd
    df = pd.DataFrame(images_info,columns=columns)
    df.to_csv('img/labels.csv')
    driver.quit()

if __name__=='__main__':
    main()