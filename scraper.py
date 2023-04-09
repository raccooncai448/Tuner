from urllib.request import urlopen
from preprocess import process
from bs4 import BeautifulSoup
import re
import string

def scrape():
    src = urlopen('https://en.wikipedia.org/wiki/Special:Random').read()

    soup = BeautifulSoup(src, 'lxml')
    title = soup.find(class_="firstHeading").text

    paras = []
    for paragraph in soup.find_all('p'):
        paras.append(str(paragraph.text))

    heads = []
    for head in soup.find_all('span', attrs={'mw-headline'}):
        heads.append(str(head.text))

    text = [val for pair in zip(paras, heads) for val in pair]
    text = ' '.join(text)
    text = re.sub(r"\[.*?\]+", '', text)
    text = text.replace('\n', '')[:-11]
    text = process(text)

    return title, text

def generate(iter, file_path,):
    print(f'Scraping for {iter} wikipedia articles')
    ref = {}
    dict = {}
    i = 0

    def encoder(input):
        return str(input.encode(encoding='utf-8').decode('ascii', 'ignore'))
    
    saved_text = ''
    dbg = 0
    while i < iter and dbg < iter*5:
        title, text = scrape()
        title, text = encoder(title), encoder(text)
        try:
            title, text = scrape()
            title, text = encoder(title), encoder(text)
            with open(file_path, "w") as f:
                f.write(text)
                saved_text += text
                saved_text += '  '
                ref[title] = text
            i += 1
        except:
            dbg += 1
            continue
    dict['text'] = saved_text
    
    if dbg == iter*5:
        raise Exception("Error with parsing wikipedia -- try re-running?")
    return dict, ref
