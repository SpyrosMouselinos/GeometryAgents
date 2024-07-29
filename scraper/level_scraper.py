import json
import os.path
import requests
import shutil
from constants import START_PAGE, SAVE_PATH_OF_SOLVABLE_LEVELS
from utils import SoupGetter, mkdir_p
from time import sleep
if os.path.exists(SAVE_PATH_OF_SOLVABLE_LEVELS):
    with open(SAVE_PATH_OF_SOLVABLE_LEVELS, 'r') as fin:
        solvable_packs = json.load(fin)
else:
    print(f"No solvable level json detected at {SAVE_PATH_OF_SOLVABLE_LEVELS}.\nYou need to run pack_scraper first!")


def scrape_level(model, level, pack, get_question=True, get_steps=True, get_explanation=True, get_images=True):
    if isinstance(level, str):
        print("Using level format of urls / strings...no way to store images!")
        get_images = False
    else:
        print("Using level format of urls / tuples... storing images!")
        level_id = level[0]
        level = level[1]
    soup = model(level)[0]
    # Get Question #
    if get_question:
        try:
            question = soup.find_all(name='div', attrs={'class': 'pi-data-value pi-font'})[0].text
        except:
            try:
                question = \
                    soup.find(name='div', attrs={'class': 'mw-parser-output'}).find_next('p').text.split(
                        'Instruction:')[-1]
                question = question.strip()
            except:
                raise Exception(f"Error at level: {level}")
            pass
    else:
        question = None

    # Get starting image #
    if get_images:
        initial_image = soup.find('figure', attrs={'class': 'pi-item pi-image'})
        if initial_image is None:
            initial_image = soup.find('figure', attrs={'class': 'thumb tnone show-info-icon'})
        initial_image = initial_image.find('a')['href']
        r = requests.get(initial_image, stream=True)  # Get request on full_url
        if r.status_code == 200:
            with open(f"../images/{pack}/{level_id}/initial.jpg", 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        else:
            print("hey")

    # Get Everything Else #
    solutions = []
    sections = soup.find_all('span', attrs={'class': 'mw-headline'})
    tutorial_exists = 0
    solution_exists = 0
    explanation_exists = 0
    for f in sections:
        if 'Tutorial' in f.text:
            tutorial_exists += 1
        if (('Solution' in f.text or 'solution' in f.text) and
                'Solutions' not in f.text and
                'Explanation' not in f.text and
                'Explanations' not in f.text and
                'From' not in f.text and
                'from' not in f.text
        ):
            solution_exists += 1
        if 'Explanation' in f.text:
            explanation_exists += 1

    if get_explanation:
        if explanation_exists > 0:
            explanation = sections[-1].find_next("p")
            if explanation:
                explanation = explanation.text
        else:
            explanation = None
    else:
        explanation = None

    if get_steps:
        tutorial = None
        if tutorial_exists > 0:
            text_sections = soup.find_all('ol')
            tutorial = text_sections[0].text

        h2 = soup.find_all('h2')
        if h2:
            h2_filtered = [f for f in h2 if 'solution' in f.text or 'Solution' in f.text]
        else:
            print(f"ISSUE ISSUE LEVEL: {level}")

        f = h2_filtered[0]
        general_comments = f.find_next('p')
        if len(general_comments) > 0:
            general_comments_h2 = general_comments.text
        else:
            general_comments_h2 = ''

        h3 = f.find_next('h3')
        steps = h3.find_next('ol')
        if steps is None:
            steps = f.find_next('ol')

        try:
            final_image = steps.find_next('figure').find_all('img')[-1]
            if 'data-src' in final_image.attrs:
                final_image = final_image['data-src']
            elif 'src' in final_image.attrs:
                final_image = final_image['src']
            else:
                print(f"Issue @ level: {level}")
            r = requests.get(final_image, stream=True)  # Get request on full_url
            if r.status_code == 200:
                with open(f"../images/{pack}/{level_id}/final.jpg", 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
            else:
                print("hey")
        except:
            print(level)
            pass

        steps = steps.text
        entry = general_comments_h2 + '\n' + steps


        if explanation is not None and explanation in entry:
            entry = entry.replace(explanation, '')

        entry = entry.replace('∠', 'the angle ')
        solutions.append(entry)
    else:
        solutions = []

    # Print them or save them! #
    return question, tutorial, solutions, explanation



model = SoupGetter(post_process_func=None)
for pack in solvable_packs:
    scrape_index = 0
    max_scrape_index = len(solvable_packs[pack])
    for level in solvable_packs[pack]:
        print(f"Scraping {scrape_index}/{max_scrape_index} levels of pack {pack}...")
        scrape_level(model, level, pack)
        scrape_index += 1
