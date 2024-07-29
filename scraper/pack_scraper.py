import json
from constants import START_PAGE, SAVE_PATH_OF_SOLVABLE_LEVELS
from utils import SoupGetter, post_process_func_pack_scraper, post_process_func_pack_solvable_identifier

url = START_PAGE
model = SoupGetter(post_process_func=post_process_func_pack_scraper)
possible_content_links = model(url)

set_of_links = possible_content_links[0]
solvable_levels = {}
model.post_process_func = post_process_func_pack_solvable_identifier
for level_link in set_of_links:
    level_name = level_link.split('/')[-1]
    model(urls=level_link, solvable_levels=solvable_levels, level_name=level_name)

with open(SAVE_PATH_OF_SOLVABLE_LEVELS, 'w') as fout:
    json.dump(solvable_levels, fout)
