from bs4 import BeautifulSoup
import requests
import pandas as pd
import sys
from datetime import datetime
import os

base = 'https://radiopaedia.org'
save_dir = 'data/articles.csv'

def get_links(link):
    """
    Returns list of article links from page linked to by LINK.
    """
    link = articles + str(i)
    source = requests.get(link).text
    soup = BeautifulSoup(source)
    return [i['href'] for i in soup.find_all('a',class_="search-result search-result-article", href=True)]

def get_example(link):
    """
    Returns article text and label of article on page LINK.
    """
    source = requests.get(link).text
    soup = BeautifulSoup(source)
    article = soup.find('div',class_="body user-generated-content")
    if not article:
    #execute this block if soup.find above returns nothing
        article = soup.find('div',class_="user-generated-content")
        #radiopaedia social media article has no body user-generated-content class
    tags = article.find_all(['p','ul'])
    text = ' '.join([tag.text for tag in tags])
    
    label_tag = soup.find('div',class_="meta-item meta-item-systems")
    if label_tag is not None:
        labels = label_tag.text.replace("System: ","")
        labels = labels.replace("Systems: " , "")
        labels = labels.replace(', ', ',')
        labels = labels.replace("Central Nervous System","CNS")
        labels = labels.replace("Head & Neck","HN")
    else:
        labels = 'Misc'
    
    return labels, text

data = {'label': [], 'text': []}

link = base + '/encyclopaedia/all/all?lang=us'

i = 0
while link is not None:
    source = requests.get(link).text
    soup = BeautifulSoup(source)
    articles = [i['href'] for i in soup.find_all('a',class_="search-result search-result-article", href=True)]
    for article in articles:
        labels, text = get_example(base + article)
        data['label'].append(labels)
        data['text'].append(text)
    
    try:
        link = base+soup.find('a',class_="next_page")['href']
    except:
        link = None
    
    i += 1
    sys.stdout.write("\rFinished page %i" % i)
    sys.stdout.flush()
    
df = pd.DataFrame(data)
if not os.path.isdir('data'):
    os.mkdir('data')
df.to_csv(save_dir, index=False)

print("COMPLETED AT FOLLOWING TIME:")
print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
