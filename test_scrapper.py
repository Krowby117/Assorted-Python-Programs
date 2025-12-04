from bs4 import BeautifulSoup
import requests

page = requests.get("https://riskofrain.fandom.com/wiki/Item_(Risk_of_Rain)")
soup = BeautifulSoup(page.text, "html.parser")

item_names = []
temp = []

#item_tables = soup.find("table", class_="article-table sortable mobile-itemTable jquery-tablesorter")
item_table = soup.find("table", class_=["article-table", "sortable", "mobile-itemTable", "jquery-tablesorter"])
#for table in item_tables:

if item_table:    
    item_links = item_table.find_all("a", href=True)

    for link in item_links:
        href = link['href']
        title = link.get('title')

        if (href.startswith('/wiki/') and title and not(title.startswith("Category:"))):
            temp.append(title)

    [item_names.append(i) for i in temp if i not in item_names]
    
else:
    print("no table found")

for item in item_names:
        print(item)