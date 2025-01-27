
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Dependencies
import numpy as np
import pandas as pd
import requests
import re
import math
import csv
from math import nan
from bs4 import BeautifulSoup
from datetime import datetime
from typing_extensions import Self

def get_completed_event_urls():

  main_url = "http://ufcstats.com/statistics/events/completed?page=all" #From this page we access all the completed events to get our data

  #Access the page with all conpleted events
  response = requests.get(main_url)

  # Check if the request was successful (status code 200)
  if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    completed_event_urls = [] #Here we will save our data

    #Loop to iterate through all the table rows on the page with all completed UFC events
    for event_name in soup.find_all('tr', class_= 'b-statistics__table-row'):
      #Access all the elements with a tag from where we need to extract the links
      urls = event_name.find('a', class_ = 'b-link b-link_style_black')

      if urls:
        url = urls.get("href") #Get the value of href
        completed_event_urls.append(url)

  else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
    return None

  return completed_event_urls

def get_fight_urls(url_range = None):
  #Scrape all the completed events http://ufcstats.com/event-details/.... (677+)
  completed_event_urls = get_completed_event_urls()

  if url_range is None:
    num_of_urls = len(completed_event_urls) #Variable to store the number of scraped cards

  else:
    completed_event_urls = completed_event_urls[url_range:]
    num_of_urls = len(completed_event_urls) #Variable to store the number of scraped cards

  i = 0 #Incrementing

  fight_urls = [] #List to store all fight urls http://ufcstats.com/fight-details/..... (7000+)

  for url in completed_event_urls:
    i = i + 1

    #Access the page with all conpleted events
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
      # Parse the HTML content of the page
      soup = BeautifulSoup(response.text, 'html.parser')

      completed_fights_urls = [] #Here we will save our data

      #Loop to iterate through all the table rows on the page with all completed UFC events
      for fights in soup.find_all('tr', class_= 'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'):
        #Access all the elements with a tag from where we need to extract the links
        urls = fights.find('a', class_ = 'b-flag b-flag_style_green')

        if urls:
          url = urls.get("href") #Get the value of href
          completed_fights_urls.append(url)

      fight_urls.extend(completed_fights_urls)
      print(f'Collection of fight urls has been completed for {i} out of {num_of_urls} events') #Just for fun

    else:
      print(f"Failed to retrieve data. Status code: {response.status_code}")
      return None

  print('Successfully collected urls for all fights')
  print('The urls are saved in the fight_urls.txt')

  file = open('fight_urls.txt','w')
  for url in fight_urls:
    file.write(url + "\n")
  file.close()

  return fight_urls

def get_fighter_urls(fight_urls):

  fighter_urls = [] #List to store the scraped data with the fighter stat pages

  num_of_urls = len(fight_urls*2)
  i = 0

  #Iterating through each fight
  for url in fight_urls:
    #Access the page
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
      # Parse the HTML content of the page
      soup = BeautifulSoup(response.text, 'html.parser')
      #Accessing all elements that contain links to the fighter's page
      fighters_urls_element = soup.find_all('a', class_ = 'b-link b-fight-details__person-link')
      #Iterate through red and blue fughter
      for element in fighters_urls_element:
        #Extract the href value from the element
        fighter_url = element.get('href')
        #Adding new urls to the list
        fighter_urls.append(fighter_url)

        i = i + 1
        print(f'{i} out of {num_of_urls} fighter urls collected') #Monitor the process

    else:
      print(f"Failed to retrieve data. Status code: {response.status_code}")
      return None

  print('Successfully collected urls for all fighters')
  print('The urls are saved in the fighter_urls.txt')

  file = open('fighter_urls.txt','w')
  for url in fighter_urls:
    file.write(url + "\n")
  file.close()

  return fighter_urls

def get_fighters_stats(fighter_urls):

  fighters_stats = []
  i = 0

  for fighter_url in fighter_urls:
    # Access the page
    response = requests.get(fighter_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
      print(fighter_url)

      # Parse the HTML content of the page
      soup = BeautifulSoup(response.text, 'html.parser')

      # Getting fighter's name
      fighter_name = soup.find('span', class_='b-content__title-highlight').text.strip()

      # Getting fighter's record
      fighter_record = soup.find('span', class_='b-content__title-record').text.replace('Record:', '').strip()
      fighter_record_values = fighter_record.split('-')
      fighter_wins = fighter_record_values[0]
      fighter_losses = fighter_record_values[1]
      fighter_draws = fighter_record_values[2] if len(fighter_record_values) > 2 else None #FIX NC

      #Getting fighter's stats
      fighter_stats_elements = soup.find_all('li', class_ = 'b-list__box-list-item b-list__box-list-item_type_block')
      fighter_stats = [stat.get_text(strip=True) for stat in fighter_stats_elements]

      #Transforming the height
      fighter_height = fighter_stats[0]
      if fighter_height != '--':
        #Assuming fighter_height is in the format 'Height:*number*\' *number*"'
        height_match = re.match(r'Height:(\d+)\' (\d+)"', fighter_height)
        if height_match is not None:
          feet, inches = map(int, height_match.groups())
        #Convert height to centimeters (1 foot = 30.48 cm, 1 inch = 2.54 cm)
        height_in_cm = (feet * 30.48) + (inches * 2.54)
      else:
        fighter_height = nan
        height_in_cm = fighter_height

      #Transforming the weight
      fighter_weight = fighter_stats[1]
      if fighter_weight != '--':
        #Assuming fighter_weight is in the format 'Weight:*number* lbs.'
        weight_match = re.match(r'Weight:(\d+) lbs\.', fighter_weight)
        if weight_match:
          weight_in_lbs = int(weight_match.group(1))
          # Convert weight to kilograms (1 lb = 0.453592 kg)
          weight_in_kg = weight_in_lbs * 0.453592
      else:
        fighter_weight = nan
        weight_in_kg = fighter_weight

      #Transforming the reach
      fighter_reach = fighter_stats[2].replace('Reach:', '').strip()
      if fighter_reach != '--':
        reach_in_inch = fighter_reach.replace('"', '').strip()
        reach_in_cm = int(reach_in_inch) * 2.54
      else:
        fighter_reach = nan
        reach_in_cm = fighter_reach

      #Transforming Date of birth
      fighter_dob = fighter_stats[4].replace('DOB:', '').strip()
      if fighter_dob != '--':
        # Convert the date of birth string to a datetime object
        dob = datetime.strptime(fighter_dob, '%b %d, %Y')
        # Get the current date
        current_date = datetime.now()
        # Calculate the age
        fighter_age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
      else:
        fighter_age = nan

      fighter_stance = fighter_stats[3].replace('STANCE:', '').strip()
      fighter_SLpM = fighter_stats[5].replace('SLpM:', '').strip()
      fighter_Str_Acc = fighter_stats[6].replace('Str. Acc.:', '').rstrip('%')
      fighter_SApM = fighter_stats[7].replace('SApM:', '').strip()
      fighter_Str_Def = fighter_stats[8].replace('Str. Def:', '').rstrip('%')
      fighter_TD_Avg = fighter_stats[10].replace('TD Avg.:', '').strip()
      fighter_TD_acc = fighter_stats[11].replace('TD Acc.:', '').rstrip('%')
      fighter_TD_def = fighter_stats[12].replace('TD Def.:', '').rstrip('%')
      fighter_Sub_Avg = fighter_stats[13].replace('Sub. Avg.:', '').strip()

      fighter_stats_dict = {
          'name': fighter_name, #Fighter name 0
          'wins': int(fighter_wins), #Number of wins by a fighter 1
          'losses': int(fighter_losses), #Number of losses by a fighter 2
          #'draws': fighter_draws, #Number of draws by a fighter (NEED TO FIX)
          #'stats': fighter_stats, #Include the list of stats in the dictionary
          'height': round(height_in_cm, 2) if not math.isnan(height_in_cm) else None, #Fighter's height 3
          'weight': round(weight_in_kg, 2) if not math.isnan(weight_in_kg) else None, #Fighter's weight 4
          'reach': round(reach_in_cm, 2) if not math.isnan(reach_in_cm) else None, #Fighter's reach 5
          'stance': fighter_stance, #Fighter's stance 6
          'age': round(float(fighter_age)) if not math.isnan(float(fighter_age)) else None, #Fighter's stance (FIX TO AGE) 7
          'SLpM': float(fighter_SLpM), #Significant Strikes Landed per Minute 8
          'sig_str_acc': float(fighter_Str_Acc)/100, #Significant Striking Accuracy 9
          'SApM': float(fighter_SApM), #Significant Strikes Absorbed per Minute 10
          'str_def': float(fighter_Str_Def)/100, #?????????????????? 11
          'td_avg': float(fighter_TD_Avg), #Average Takedowns Landed per 15 minutes 12
          'td_acc': float(fighter_TD_acc)/100, #Takedown Accuracy 13
          'td_def': float(fighter_TD_def)/100, #Takedown Defense (the % of opponents TD attempts that did not land) 14
          'sub_avg': float(fighter_Sub_Avg), #Average Submissions Attempted per 15 minutes 15
      }

      fighters_stats.append(fighter_stats_dict)

      #Check the process
      i = i + 1
      print(i, "out of", len(fighter_urls))

      # Save the result to a text file on each iteration
      with open('/content/fighters_stats.txt', 'a') as file:
        for key, value in fighter_stats_dict.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
      print('Data has been saved to the file\n')
      print(fighter_stats_dict)

  return fighters_stats

def get_red_fighters_stats(fighters_stats):

  red_fighters_stats = []
  for index, fighter in enumerate(fighters_stats):
      if index % 2 == 0:  # Even index, blue fighter
          red_fighters_stats.append(fighter)
  #print(red_fighters_stats)
  return red_fighters_stats

def get_blue_fighters_stats(fighters_stats):

  blue_fighters_stats = []
  for index, fighter in enumerate(fighters_stats):
      if index % 2 != 0:
          blue_fighters_stats.append(fighter)
  #print(blue_fighters_stats)
  return blue_fighters_stats

# Function to creat a list of dictionaries with the physical and career stats for the red fighters
def create_r_fighter_dicts(red_fighters_stats):
    red_fighter_dicts = []
    for red_fighter_stat in red_fighters_stats:
        red_fighter_dict = {
            'r_wins_total': red_fighter_stat['wins'],
            'r_losses_total': red_fighter_stat['losses'],
            'r_age': red_fighter_stat['age'],
            'r_height': red_fighter_stat['height'],
            'r_weight': red_fighter_stat['weight'],
            'r_reach': red_fighter_stat['reach'],
            'r_stance': red_fighter_stat['stance'],
            'r_SLpM_total': red_fighter_stat['SLpM'],
            'r_SApM_total': red_fighter_stat['SApM'],
            'r_sig_str_acc_total': red_fighter_stat['sig_str_acc'],
            'r_td_acc_total': red_fighter_stat['td_acc'],
            'r_str_def_total': red_fighter_stat['str_def'],
            'r_td_def_total': red_fighter_stat['td_def'],
            'r_sub_avg': red_fighter_stat['sub_avg'],
            'r_td_avg': red_fighter_stat['td_avg']
        }
        red_fighter_dicts.append(red_fighter_dict)
    #print(red_fighter_dicts)
    return red_fighter_dicts

# Function to create a list of dictionaries with the physical and career stats for the blue fighters
def create_b_fighter_dicts(blue_fighters_stats):
    blue_fighter_dicts = []
    for blue_fighter_stat in blue_fighters_stats:
        blue_fighter_dict = {
            'b_wins_total': blue_fighter_stat['wins'],
            'b_losses_total': blue_fighter_stat['losses'],
            'b_age': blue_fighter_stat['age'],
            'b_height': blue_fighter_stat['height'],
            'b_weight': blue_fighter_stat['weight'],
            'b_reach': blue_fighter_stat['reach'],
            'b_stance': blue_fighter_stat['stance'],
            'b_SLpM_total': blue_fighter_stat['SLpM'],
            'b_SApM_total': blue_fighter_stat['SApM'],
            'b_sig_str_acc_total': blue_fighter_stat['sig_str_acc'],
            'b_td_acc_total': blue_fighter_stat['td_acc'],
            'b_str_def_total': blue_fighter_stat['str_def'],
            'b_td_def_total': blue_fighter_stat['td_def'],
            'b_sub_avg': blue_fighter_stat['sub_avg'],
            'b_td_avg': blue_fighter_stat['td_avg']
        }
        blue_fighter_dicts.append(blue_fighter_dict)
    #print(blue_fighter_dicts)
    return blue_fighter_dicts

#Function to automatically create dictionaries for statistical info on the fight page
def create_stats_dict(current_fight_stats):
  print("Length of current_fight_dict:", len(current_fight_stats))  # Debugging statement
  print("current_fight_dict:", current_fight_stats)  # Debugging statement

  #Calculating extra stats for the red fighter
  if len(current_fight_stats) >= 11:
    r_sig_str_values = current_fight_stats[4].split(' of ') # Splitting 'r_sig_str' and 'r_sig_str_att' values
    r_total_str_values = current_fight_stats[8].split(' of ') # Splitting 'r_total_str' and 'r_total_str_att' values
    r_td_values = current_fight_stats[10].split(' of ') # Splitting 'r_td' and 'r_td_att' values
    try:
      r_str_acc = (int(r_total_str_values[0])/int(r_total_str_values[1]))*100 if r_total_str_values != '---' else 0 #Calculating the total striking accuracy for the red fighter
    except ZeroDivisionError:
      # Handle the case where the divisor is zero
      r_str_acc = 0
    r_ctrl_time = current_fight_stats[18]
    if ':' not in r_ctrl_time:
      # Handle the case where the format is not as expected
      # You can choose to set r_ctrl_time_sec to a default value or handle it in another way
      r_ctrl_time_sec = 0
    else:
      minutes, seconds = map(int, r_ctrl_time.split(':'))
      r_ctrl_time_sec = minutes * 60 + seconds

    #Calculating extra stats for the blue fighter
    b_sig_str_values = current_fight_stats[5].split(' of ') # Splitting 'b_sig_str' and 'b_sig_str_att' values
    b_total_str_values = current_fight_stats[9].split(' of ') # Splitting 'b_total_str' and 'b_total_str_att' values
    b_td_values = current_fight_stats[11].split(' of ') # Splitting 'b_td' and 'b_td_att' values
    try:
      b_str_acc = (int(b_total_str_values[0])/int(b_total_str_values[1]))*100 if b_total_str_values != '---' else 0 #Calculating the total striking accuracy for the bed fighter
    except ZeroDivisionError:
      # Handle the case where the divisor is zero
      b_str_acc = 0
    b_ctrl_time = current_fight_stats[19]
    if ':' not in b_ctrl_time:
      # Handle the case where the format is not as expected
      # You can choose to set r_ctrl_time_sec to a default value or handle it in another way
      b_ctrl_time_sec = 0
    else:
      minutes, seconds = map(int, b_ctrl_time.split(':'))
      b_ctrl_time_sec = minutes * 60 + seconds

    #Creating variables for dictionary assignment
    r_kd = round(float(current_fight_stats[2])) #Knockdown by red 0
    r_sig_str = round(float(r_sig_str_values[0])) #Significant strkes landed by red 1
    r_sig_str_att = round(float(r_sig_str_values[1])) #Significant strkes attempted by red 2
    r_sig_str_acc = current_fight_stats[6].rstrip('%') if current_fight_stats[6] != '---' else 0 #Significant strke accuracy by red 3
    r_str = round(float(r_total_str_values[0])) #Total strikes landed by red 4
    r_str_att = round(float(r_total_str_values[1])) #Total strikes attempted by red 5
    r_str_acc = round(r_str_acc)/100 if r_str_acc != '---' else 0 #Total strikes accuracy by red 6
    r_td = round(float(r_td_values[0])) #Takedowns landed by red 7
    r_td_att = round(float(r_td_values[1])) #Takedowns attempted by red 8
    r_td_acc = current_fight_stats[12].rstrip('%') if current_fight_stats[12] != '---' else 0 #Takedowns accuracy by red 9
    r_sub_att = round(float(current_fight_stats[14])) #Submission attempted by red 10
    r_rev = round(float(current_fight_stats[16])) #No idea what that means (Reverse????) 11
    r_ctrl = r_ctrl_time_sec #Control time by red 12

    b_kd = round(float(current_fight_stats[3]))  #Knockdown by blue
    b_sig_str = round(float(b_sig_str_values[0])) #Significant strikes landed by blue
    b_sig_str_att = round(float(b_sig_str_values[1]))  #Significant strikes attempted by blue
    b_sig_str_acc = current_fight_stats[7].rstrip('%') if current_fight_stats[7] != '---' else 0  #Significant strike accuracy by blue
    b_str = round(float(b_total_str_values[0]))  #Total strikes landed by blue
    b_str_att = round(float(b_total_str_values[1]))  #Total strikes attempted by blue
    b_str_acc = round(b_str_acc)/100 if b_str_acc != '---' else 0  #Total strikes accuracy by blue
    b_td = round(float(b_td_values[0]))  #Takedowns landed by blue
    b_td_att = round(float(b_td_values[1]))  #Takedowns attempted by blue
    b_td_acc = current_fight_stats[13].rstrip('%') if current_fight_stats[13] != '---' else 0  #Takedowns accuracy by blue
    b_sub_att = round(float(current_fight_stats[15]))  #Submission attempted by blue
    b_rev = round(float(current_fight_stats[17]))  #No idea what that means (Reverse????)
    b_ctrl = b_ctrl_time_sec  #Control time by blue

    #Creating a current_fight_dict for total data
    totals_dict = {
      #RED CURRENT FIGHT STATS
      'r_kd': r_kd, #Knockdown by red 0
      'r_sig_str': r_sig_str, #Significant strkes landed by red 1
      'r_sig_str_att': r_sig_str_att, #Significant strkes attempted by red 2
      'r_sig_str_acc': float(r_sig_str_acc)/100, #Significant strke accuracy by red 3
      'r_str': r_str, #Total strikes landed by red 4
      'r_str_att': r_str_att, #Total strikes attempted by red 5
      'r_str_acc': r_str_acc, #Total strikes accuracy by red 6
      'r_td': r_td, #Takedowns landed by red 7
      'r_td_att':r_td_att, #Takedowns attempted by red 8
      'r_td_acc': float(r_td_acc)/100, #Takedowns accuracy by red 9
      'r_sub_att': r_sub_att, #Submission attempted by red 10
      'r_rev': r_rev, #No idea what that means (Reverse????) 11
      'r_ctrl_sec': r_ctrl, #Control time by red 12

      #RED CAREER STATS (Accumulated)
        #Current streak
        #Longest win streak
        #Longest lose streak
        #Number of draws
        #Number of losses
        #Number of wins
        #Number of wins by DM
        #Number of wins by KO/TKO
        #Number of wins by Unanimous decision
        #Number of wins by Split decision
        #Number of wins by Majority decision
        #Number of wins by Submissions
        #Number of wins by Doctor Stoppage

        #Number of rounds fought
        #Number of title fights

        #Average significant strikes landed
        #Average significant strikes attempted
        #Average significant strikes acc
        #Average submission attempts
        #Average strikes landed
        #Average strikes attempted
        #Average strikes acc
        #Average takedowns landed
        #Average takedown attempts
        #Average takedown acc

      #BLUE
      'b_kd': b_kd,  #Knockdown by blue
      'b_sig_str': b_sig_str,  #Significant strikes landed by blue
      'b_sig_str_att': b_sig_str_att,  #Significant strikes attempted by blue
      'b_sig_str_acc': float(b_sig_str_acc)/100,  #Significant strike accuracy by blue
      'b_str': b_str,  #Total strikes landed by blue
      'b_str_att': b_str_att,  #Total strikes attempted by blue
      'b_str_acc': b_str_acc,  #Total strikes accuracy by blue
      'b_td': b_td,  #Takedowns landed by blue
      'b_td_att': b_td_att,  #Takedowns attempted by blue
      'b_td_acc': float(b_td_acc)/100,  #Takedowns accuracy by blue
      'b_sub_att': b_sub_att,  #Submission attempted by blue
      'b_rev': b_rev,  #No idea what that means (Reverse????)
      'b_ctrl_sec': b_ctrl,  #Control time by blue


      # #DIFFS !!!ALL THE DIFFERENCES CALCULATED AS (RED - BLUE)!!!
      # 'kd_diff':  r_kd - b_kd,  #Knockdown difference
      # 'sig_str_diff': r_sig_str - b_sig_str, #Significant strikes landed difference
      # 'sig_str_att_diff': r_sig_str_att - b_sig_str_att, #Significant strikes attempted difference
      # 'sig_str_acc_diff': float(r_sig_str_acc) / 100 - float(b_sig_str_acc) / 100, #Significant strikes accuracy difference
      # 'str_diff': r_str - b_str,  #Total stikes landed difference
      # 'str_att_diff': r_str_att - b_str_att,  #Total strikes attempted difference
      # 'str_acc_diff': float(r_str_acc)/100 - float(b_str_acc)/100, #Total strikes accuracy difference
      # 'td_diff': r_td - b_td,  #Takedowns landed difference
      # 'td_att_diff': r_td_att - b_td_att,  #Takedowns attempted difference
      # 'td_acc_diff': float(r_td_acc)/100 - float(b_td_acc)/100,  #Takedowns accuracy difference
      # 'sub_att_diff': r_sub_att - b_sub_att,  #Submission attempts difference
      # 'rev_diff': r_rev - b_rev,  #Rev difference
      # #'ctrl_diff': r_ctrl - b_ctrl,  #Control time difference

      # 'age_diff': nan, #Age difference
      # 'height_diff': nan, #Height difference
      # 'reach_diff': nan,  #Reach difference
      # 'stance_diff': nan,  #Stance difference
      # 'weight_diff': nan,  #Weight difference

      # 'SLPN_diff': nan, # Significant Strikes Landed per Minute
      # 'SApM_diff': nan, # Significant Strikes Absorbed per Minute
      # 'sig_str_def_diff': nan, # Significant Strike Defence


        # Current streak difference
        #Longest win streak difference
        #Longest lose streak difference
        #Number of draws difference
        #Number of losses difference
        #Number of wins difference
        #Number of rounds fought difference
        #Number of title fights difference

        #Average significant strikes landed difference
        #Average significant strikes attempted difference
        #Average significant strikes acc difference
        #Average submission attempts difference
        #Average strikes landed difference
        #Average strikes attempted difference
        #Average strikes acc difference
        #Average takedowns landed difference
        #Average takedown attempts difference
        #Average takedown acc difference
        }

  else:
    #Creating an empty dictionary in case we can't get the data
    totals_dict = {
      #RED CURRENT FIGHT STATS
      'r_kd': nan, #Knockdown by red 0
      'r_sig_str': nan, #Significant strkes landed by red 1
      'r_sig_str_att': nan, #Significant strkes attempted by red 2
      'r_sig_str_acc': nan, #Significant strke accuracy by red 3
      'r_total_str': nan, #Total strikes landed by red 4
      'r_total_str_att': nan, #Total strikes attempted by red 5
      'r_total_str_acc': nan, #Total strikes accuracy by red 6
      'r_td': nan, #Takedowns landed by red 7
      'r_td_att': nan, #Takedowns attempted by red 8
      'r_td_acc': nan, #Takedowns accuracy by red 9
      'r_sub_att': nan, #Submission attempted by red 10
      'r_rev': nan, #No idea what that means (Reverse????) 11
      'r_ctrl': nan, #Control time by red 12

      #BLUE
      'b_kd': nan,  #Knockdown by blue
      'b_sig_str': nan,  #Significant strikes landed by blue
      'b_sig_str_att': nan,  #Significant strikes attempted by blue
      'b_sig_str_acc': nan,  #Significant strike accuracy by blue
      'b_total_str': nan,  #Total strikes landed by blue
      'b_total_str_att': nan,  #Total strikes attempted by blue
      'b_total_str_acc': nan,  #Total strikes accuracy by blue
      'b_td': nan,  #Takedowns landed by blue
      'b_td_att': nan,  #Takedowns attempted by blue
      'b_td_acc': nan,  #Takedowns accuracy by blue
      'b_sub_att': nan,  #Submission attempted by blue
      'b_rev': nan,  #No idea what that means (Reverse????)
      'b_ctrl': nan,  #Control time by blue

      # #DIFFS !!!ALL THE DIFFERENCES CALCULATED AS (RED - BLUE)!!!
      # 'kd_diff': nan,  #Knockdown difference
      # 'sig_str_diff': nan, #Significant strikes landed difference
      # 'sig_str_att_diff': nan, #Significant strikes attempted difference
      # 'sig_str_acc_diff': nan, #Significant strikes accuracy difference
      # 'total_str_diff': nan,  #Total stikes landed difference
      # 'total_str_att_diff': nan,  #Total strikes attempted difference
      # 'total_str_acc_diff': nan, #Total strikes accuracy difference
      # 'td_diff': nan,  #Takedowns landed difference
      # 'td_att_diff': nan,  #Takedowns attempted difference
      # 'td_acc_diff': nan,  #Takedowns accuracy difference
      # 'sub_att_diff': nan,  #Submission attempts difference
      # 'rev_diff': nan,  #Rev difference
      # 'ctrl_diff': nan,  #Control time difference

      # 'age_diff': nan, #Age difference
      # 'height_diff': nan, #Height difference
      # 'reach_diff': nan,  #Reach difference
      # 'stance_diff': nan,  #Stance difference
      # 'weight_diff': nan,  #Weight difference

      # 'SLPN_diff': nan, # Significant Strikes Landed per Minute
      # 'SApM_diff': nan, # Significant Strikes Absorbed per Minute
      # 'sig_str_def_diff': nan, # Significant Strike Defence
  }

  return totals_dict

def create_common_dict(soup):

    #Getting event name
    event_name = soup.find('h2', class_ = 'b-content__title').text.strip()

    #Getting fighter's name
    fighters = soup.find_all('h3', class_='b-fight-details__person-name')
    fighter_names = []
    for fighter in fighters:
      name = fighter.get_text(strip=True)
      fighter_names.append(name)

    #Getting fighter's status (WIN or LOSE)
    statuses = soup.find_all('i', class_ = 'b-fight-details__person-status')
    fighter_statuses = []
    for s in statuses:
      status = s.get_text(strip=True)
      fighter_statuses.append(status)

    #Calculate who is the winner
    winner = []
    for fs in fighter_statuses[0]:
      if fs == 'W':
        winner = 'Red'
      elif fs == 'L':
        winner = 'Blue'

    #Getting the fight's title
    fight_title = soup.find('i', class_='b-fight-details__fight-title')
    if fight_title is not None:
        fight_title = fight_title.text.strip()
    else:
        # Handle case where the element is not found
        fight_title = nan

    #Getting general statistics about the fight
    method = soup.find('i', class_ = 'b-fight-details__text-item_first').text.replace('Method:', '').strip()
    gen_stats = soup.find_all('i', class_='b-fight-details__text-item')
    fight_data = []
    for s in gen_stats:
      stat = s.get_text(strip=True)
      fight_data.append(stat)

    #Calculating if the fight is title bout or no
    is_title_bout = 0
    if 'Title' in fight_title:
      is_title_bout = 1
    else:
      is_title_bout = 0

    #Calculate gender
    gender = 0
    if "Women's" in fight_title:
      gender = 'Women'
    else:
      gender ='Men'

    # Calculate total number of rounds
    total_rounds_text = fight_data[2].replace('Time format:', '')
    match = re.search(r"(\d+)", total_rounds_text)
    if match:
        total_rounds = int(match.group(1))
    else:
        total_rounds = None

    # Calculate the time of the fight in seconds
    fight_time = fight_data[1].replace('Time:', '')
    minutes, seconds = map(int, fight_time.split(':'))
    total_seconds = minutes * 60 + seconds

    common_dict = {
    'event_name': str(event_name),
    'r_fighter': fighter_names[0],
    'b_fighter': fighter_names[1],
    'winner': winner,
    'weight_class': fight_title.split(' Bout')[0].strip(),
    'is_title_bout': is_title_bout,
    'gender': gender,
    'method': method,
    'finish_round': int(fight_data[0].replace('Round:','')),
    'total_rounds': total_rounds,
    'time_sec': total_seconds,
    'referee': fight_data[3].replace('Referee:',''),
    #'details':
    }

    return common_dict

def get_fight_data(fight_urls):

  total_page_dicts = [] #Variable to store all the scraped data from the card

  i = 0
  n = len(fight_urls)

  for fight_url in fight_urls:

    response = requests.get(fight_url)

    if response.status_code == 200:

      soup = BeautifulSoup(response.text, 'html.parser')

      #Get the links for fighters in each fight of the card
      #fighter_details_links = get_fighter_urls(fight_url) RAISES AN ERROR
      #print(fighter_details_links)
      #Get the data for both fighters
      #fighters_stats = get_fighters_stats(fighter_urls)
      #Split the fighters data into red and blue
      #red_fighters_stats = get_red_fighters_stats(fighters_stats) #Personal stats for the RED fighter (at the current moment)
      #CREATE A DICTIONARY FOR THE RED FIGHTER
      #blue_fighters_stats = get_blue_fighters_stats(fighters_stats) #Personal stats for the BLUE fighter (at the current moment)
      #CREATE A DICTIONARY FOR THE BLUE FIGHTER

      #Getting the common data for the fight
      common_dict = create_common_dict(soup)

      #Getting total stats about the fight
      tables = soup.find_all('table', {'style': 'width: 745px'})
      stats_data = []
      #print(tables)
      for t in tables:
        stats = soup.find_all('p', class_ = 'b-fight-details__table-text')
        for s in stats:
          stat = s.text.strip()
          stats_data.append(stat)

      # #You can add the data from here to the total dictionary if you want to perform in depth analysis
      # round_1 = stats_data[20:40] #Data for the 1st round
      # round_2 = stats_data[40:60] #Data for the 2nd round
      # round_3 = stats_data[60:80] #Data for the 3rd round
      # round_4 = stats_data[80:100] #Data for the 4th round
      # round_5 = stats_data[100:120] #Data for the 5th round

      # #You can add the data from here to the total dictionary if you want to perform in depth analysis
      # sig_strikes = stats_data[120:138] #Data for the whole fight significant strikes
      # sig_strikes_r1 = stats_data[138:156] #Data for the 1st round significant strikes
      # sig_strikes_r2 = stats_data[156:174] #Data for the 2nd round significant strikes
      # sig_strikes_r3 = stats_data[174:192] #Data for the 3rd round significant strikes
      # sig_strikes_r4 = stats_data[192:210] #Data for the 4th round significant strikes
      # sig_strikes_r5 = stats_data[210:228] #Data for the 5th round significant strikes

      if stats_data:
        current_fight_stats = stats_data[0:20] #Data from the "Totals" table

        #Create total dictionary
        totals_dict = create_stats_dict(current_fight_stats)

        #Combine all the dictionaries
        page_dict = {**common_dict, **totals_dict}
        #Add the combined dictionaries to the list with all data
        total_page_dicts.append(page_dict)

        print(f'current url:{fight_url}')
        print(f'{i} of {n} completed')
        i = i + 1
      else:
        print(f"No stats data found for {fight_url}. Skipping...")

        stats_data = []

        #Create total dictionary
        totals_dicts = create_stats_dict(stats_data)

        #Combine all the dictionaries
        page_dict = {**common_dict, **totals_dict}

        print(f'Current dict:{page_dict}')

        total_page_dicts.append(page_dict)

    else:
      print(f"Failed to retrieve data. Status code: {response.status_code}")
      return None

  return total_page_dicts

def combine_fight_and_personal_stats(total_page_dicts, red_fighters_dicts, blue_fighters_dicts):

  total_page_dict_df = pd.DataFrame(total_page_dicts)
  red_fighters_dicts_df = pd.DataFrame(red_fighters_dicts)
  #print(red_fighters_dicts_df.info())
  blue_fighters_dicts_df = pd.DataFrame(blue_fighters_dicts)
  #print(blue_fighters_dicts_df.info())

  column_index = 'r_ctrl' #Define the split column for distinguishing fighters
  column_index2 = 'b_ctrl' #Define the split index for diff
  column_index_position = 24  #Find the index of the column in total_page_dict_df
  column_index_position2 = 14 #Find the index of the column in total_page_dicts_split2

  # Split df1 into two parts: before and after the specified column
  total_page_dicts_split_red = total_page_dict_df.iloc[:, :column_index_position + 1]
  total_page_dicts_split_bluediff = total_page_dict_df.iloc[:, column_index_position + 1:]
  total_page_dicts_split_blue = total_page_dicts_split_bluediff.iloc[:, :column_index_position2 + 1]
  total_page_dicts_split_diff = total_page_dicts_split_bluediff.iloc[:, column_index_position2 + 1:]
  #Concatenate df1_before, df2, and df1_after horizontally
  halfresult = pd.concat([total_page_dicts_split_red, red_fighters_dicts_df], axis=1)
  halfresult2 = pd.concat([halfresult, total_page_dicts_split_blue, blue_fighters_dicts_df], axis=1)
  result = pd.concat([halfresult2, total_page_dicts_split_diff], axis = 1)

  return result

def calculate_diff(df):
    columns_to_diff = [
        'kd', 'sig_str', 'sig_str_att', 'sig_str_acc',
        'str', 'str_att', 'str_acc', 'td',
        'td_att', 'td_acc', 'sub_att', 'rev',
        'ctrl_sec', 'wins_total', 'losses_total',
        'age', 'height', 'weight', 'reach', 'SLpM_total', 'SApM_total',
        'sig_str_acc_total', 'td_acc_total',
        'str_def_total', 'td_def_total', 'sub_avg', 'td_avg'
    ]

    for col in columns_to_diff:
        df[f'{col}_diff'] = df[f'r_{col}'] - df[f'b_{col}']

    return df

def create_large_dataset(url_range = None):
  fight_urls = get_fight_urls(url_range)
  fighter_urls = get_fighter_urls(fight_urls)
  fighters_stats = get_fighters_stats(fighter_urls)
  red_fighters_stats = get_red_fighters_stats(fighters_stats)
  blue_fighters_stats = get_blue_fighters_stats(fighters_stats)
  red_fighters_dicts = create_r_fighter_dicts(red_fighters_stats)
  blue_fighters_dicts = create_b_fighter_dicts(blue_fighters_stats)
  total_page_dicts = get_fight_data(fight_urls)
  full_fight_data = combine_fight_and_personal_stats(total_page_dicts, red_fighters_dicts, blue_fighters_dicts)
  full_fight_data = calculate_diff(full_fight_data)
  completed_events_large_df = pd.DataFrame(full_fight_data)
  completed_events_large_df.to_csv('completed_events_large.csv', index=False)
  print('Large dataset has been collected you can access it in the completed_events_large.csv file')
  print('All fight urls can be found in fight_urls.txt')
  print('All fighter urls can be found in the fighter_stats.txt')

  return completed_events_large_df

# Takes a few hours to create so use with caution

large_dataset = create_large_dataset()