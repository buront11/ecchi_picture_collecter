import time
import re
import csv
import json
import urllib
import math
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

PROXY = {
        'http':'http://proxy.nagaokaut.ac.jp:8080',
        'https':'http://proxy.nagaokaut.ac.jp:8080'
    }

HEADER = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"
        }

class ImageTypeError(Exception):
    """画像のtypeが選択されていないことを知らせる例外クラス"""
    pass

def download(uri, save_path):
    print("Downloading from " + uri)
    proxy_support = urllib.request.ProxyHandler(PROXY)
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)
    request = urllib.request.Request(uri, headers=HEADER) 
    img_data = urllib.request.urlopen(request).read()
    uri_list = uri.split("/")
    file_name = uri_list[len(uri_list) - 1]
    with open(save_path + "/" + file_name, mode="wb") as f:
        f.write(img_data)
        print("Saved to " + save_path + "/" + file_name + ".")

def fetch(data, save_path, cnt, total, limit):
    if limit and cnt > limit:
        return
    for value in data:
        if limit and cnt > limit:
            break
        image = value["webformatURL"] # imageURL(オリジナルサイズ), webformatURL(640px), largeImageURL(1280px), fullHDURL(1920px)などがある。
        print(str(cnt) + "/" + str(total))
        cnt = cnt + 1
        time.sleep(3)
        download(image, save_path)
    return cnt

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

def ecchi():
    # Seleniumをあらゆる環境で起動させるChromeオプション
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")
    options.add_argument('--start-maximized')
    # options.add_argument('--headless')

    # ブラウザの起動
    driver = webdriver.Chrome(executable_path='./chromedriver',chrome_options=options)

    driver.get("https://nijie.info/login.php")
    print('Email : ', end='')
    login_user = 'manmami.a.mario@gmail.com'
    print('Password : ', end='')
    login_pass = 'yosiki2227'

    # 各タグごとの持ってくる画像の上限枚数
    image_limit = 1000

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
        url = "http://nijie.info/search.php?type=&word="+search_tag+"&p=1&mode=&illust_type=1&sort=2&pd=&con="
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
            # ベスト３作品を排除
            for url in urls[:-3]:
                time.sleep(random.randrange(5,10))
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
            try:
                element = driver.find_element_by_class_name('right').click()
            except:
                break
        # elements = driver.find_elements_by_class_name("clearfix")
        # for element in elements:
        #     img_tags = element.find_elements_by_tag_name('img')
        #     print(img_tags)
        #     dd
    df = pd.DataFrame(images_info,columns=columns)
    df.to_csv('img/untreated_labels.csv', index=False)
    driver.quit()

def no_ecchi():
    api_key = "23096309-fb0bf9c1e4635b19578c1362c" #PixabayのAPIキーを記述

    print("Pixabayから画像をダウンロードするプログラム。")
    with open('no_ecchi_tag.csv', 'r') as f:
        reader = csv.reader(f)
        keywords = [row for row in reader][0]

    image_limit = 800
    each_images = image_limit//len(keywords)
    utils.mkdir('./no_ecchi_img', delete=True)

    for keyword in keywords:

        page = 1
        per_page = 200
        limit = each_images # ここで設定した枚数までしかダウンロードしない。0の場合は無制限。

        uri = "https://pixabay.com/api/"

        prms = {
            "key" : api_key,
            "q" : keyword,
            "lang" : "ja", #デフォルトはen
            "image_type" : "all", #all, photo, illustration, vector
            "orientation" : "all", #all, horizontal, vertical
            "category" : "", # fashion, nature, backgrounds, science, education, people, "feelings, religion, health, places, animals, industry, food, computer, sports, transportation, travel, buildings, business, music
            "min_width" : "0", # 最小の横幅
            "min_height" : "0", # 最小の立幅
            "colors" : "", # "grayscale", "transparent", "red", "orange", "yellow", "green", "turquoise", "blue", "lilac", "pink", "white", "gray", "black", "brown"
            "editors_choice" : "false", # Editor's Choiceフラグが立ったものに限定したい場合はtrue
            "safesearch" : "false", # セーフサーチ false or true
            "order" : "popular", # 並び順（popular or latest）
            "page" : page, # デフォルトは1、ページネーションのページ番号らしい
            "per_page" : per_page, # デフォルトは20。1ページあたりの表示件数。3〜200まで
            "callback" : "", # JSONPのコールバック関数を指定できるらしい
            "pretty" : "false", # JSON出力をインデントするかどうか false or true
        }

        save_path = "./no_ecchi_img"
        print("save_path:" + save_path)

        req = requests.get(uri, params=prms, proxies=PROXY)
        result = req.json()

        total = result["totalHits"]
        cnt = 1
        cnt = fetch(result["hits"], save_path, cnt, total, limit)

        page_num = math.ceil(total / per_page)

        # ページネーション対応
        if page_num > page:
            for i in range(page + 1, page_num + 1, 1):
                if limit and cnt > limit:
                    break
                prms["page"] = i
                print(i)
                req = requests.get(uri, params=prms)
                result = req.json()
                cnt = fetch(result["hits"], save_path, cnt, total, limit)

    print("Download Finished.")

if __name__=='__main__':
    no_ecchi()