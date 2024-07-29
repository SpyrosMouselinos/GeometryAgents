import json
import sys
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import errno, os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from constants import BASE_PAGE, START_PAGE, SAVE_PATH_OF_SOLVABLE_LEVELS


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def clean_level_pack_links(set_of_results):
    final_links = []
    for f in set_of_results:
        if len(f) == 0 or len(f) > 1:
            continue
        final_links.append(BASE_PAGE + f[0]['href'])
    return final_links


def clean_level_pack_level_links(set_of_results):
    final_links = []
    for f in set_of_results:
        if len(f) == 0:
            continue
        elif len(f) > 1:
            f = [f[-1]]
        final_links.append(BASE_PAGE + f[0]['href'])
    return final_links


def post_process_func_pack_scraper(soup):
    content = soup.find("table")
    possible_content_links = [f.find_all(name='a') for f in content.find_all(name='th')]
    result_obj = clean_level_pack_links(possible_content_links)
    return result_obj


def post_process_func_pack_solvable_identifier(level_name, solvable_levels, soup):
    content = soup.find('table', {'class': 'article-table'})
    possible_content_links = [f.find_all('a') for f in content.find_all('td')]
    clean_possible_links = clean_level_pack_level_links(possible_content_links)
    if len(clean_possible_links) > 0:
        solvable_levels.update({level_name: clean_possible_links})
    return solvable_levels


class SoupGetter:
    def __init__(self, post_process_func=None):
        if post_process_func is None:
            post_process_func = lambda soup, *args: soup
        self.post_process_func = post_process_func
        self.driver_options = webdriver.ChromeOptions()
        self.driver_options.add_argument("disable-infobars")
        self.driver_options.add_argument('--disable-extensions')
        self.driver_options.add_argument('--headless')
        self.driver_options.add_argument('--disable-gpu')
        self.driver_options.add_argument('--disable-dev-shm-usage')
        self.driver_options.add_argument('--ignore-certificate-errors')
        self.driver_options.add_argument('--no-sandbox')
        self.driver = None

    def load_selenium(self):
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.driver_options)

    def unload_selenium(self):
        del self.driver
        self.driver = None

    def make_request(self, urls, **kwargs):
        if isinstance(urls, list):
            pass
        else:
            urls = [urls]

        outputs = []

        for url in urls:
            try:
                req = requests.get(url, allow_redirects=False)
            except Exception as e:
                print("Error Loading the page", e)
                sys.exit(1)

            if self.driver:
                self.driver.get(url)
                content = self.driver.page_source
            else:
                content = req.content

            if content != b'':
                soup = BeautifulSoup(content, "html.parser")
                output = self.post_process_func(soup=soup, **kwargs)
                outputs.append(output)
            else:
                print(f'Empty Content @ {url}, check your chromedriver!')
                sys.exit(1)
        return outputs

    def __call__(self, urls, **kwargs):
        return self.make_request(urls, **kwargs)


def enumerate_solvable_levels():
    """
    Run this once to convert entries of solvable_levels.json from list, to list of tuples
    """
    if os.path.exists(SAVE_PATH_OF_SOLVABLE_LEVELS):
        with open(SAVE_PATH_OF_SOLVABLE_LEVELS, 'r') as fin:
            data = json.load(fin)

        for k, v in data.items():
            if isinstance(v, list):
                if isinstance(v[0], str):
                    print("File is in format List[urls]...converting")
                    convert_flag = True
                    break
                else:
                    print("File is in format List[Tuple(int,url)]...all good")
                    convert_flag = False
                    break
            else:
                print("File is not correct, please re-run pack_scraper.py!")

        if convert_flag:
            new_data = {}
            for k, v in data.items():
                new_data.update({k: [(i, v) for i, v in enumerate(v)]})
            with open('test.json', 'w') as fout:
                json.dump(new_data, fout)
    return
