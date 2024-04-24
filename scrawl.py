import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_player_values(year, headers=None):
    """
    Fetch player names and their corresponding values from a given URL.
    
    Parameters:
    - url (str): The URL to fetch data from.
    - headers (dict, optional): A dictionary of HTTP headers to send with the request.

    Returns:
    - dict: A dictionary of player names and their corresponding values.
    - None: If the request fails or data cannot be retrieved.
    """

    url = 'https://hoopshype.com/nba2k/'+str(year-1)+'-'+str(year) +'/'
    if headers is None:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        player_values = {}
        
        rows = soup.find_all('tr')
        for row in rows:
            name_tag = row.find('td', class_='name')
            value_tag = row.find('td', class_='value')
            
            if name_tag and value_tag:
                name = name_tag.text.strip()
                value = value_tag.text.strip()
                
                player_values[name] = value
        
        # Remove any unwanted header entries if exists
        player_values.pop('Player', None)  # Safely remove 'Player' key if it exists
        
        return player_values
    else:
        print("Failed to retrieve data")
        return None

# Example usage
# year = 2023
# player_stats = fetch_player_values(year)
# if player_stats:
#     print(player_stats)


def fetch_nba_player_stats(year):
    """
    Fetch NBA player statistics from a given URL and return a dictionary with player names as keys.
    
    Args:
    url (str): URL of the website containing NBA player stats in a table format.
    
    Returns:
    dict: Each key is a player's name and the value is another dictionary of that player's stats.
    """
    # Send HTTP request to the URL
    url = 'https://www.basketball-reference.com/leagues/NBA_'+str(year)+'_totals.html'
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code != 200:
        return "Failed to retrieve the webpage."

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table - assumes the data is in the first table
    table = soup.find('table')
    
    # Extract column headers
    headers = [th.text for th in table.find('thead').find_all('th')[1:]]  # Skip the first header (usually 'Rk' or rank)
    
    # Dictionary to hold all player data
    player_data = {}

    # Iterate over each row of the table body
    for row in table.find('tbody').find_all('tr'):
        # Extract each cell of the row
        cells = row.find_all('td')
        if cells:
            # Extract player name which is usually the first cell in the row with a link
            player_name = cells[0].text.strip()
            stats = {}
            # Start from 1 because 0 is player name which we're using as the key
            for index, cell in enumerate(cells[1:]):  # Start from second element to skip player name
                if index < len(headers):  # Ensure no index errors
                    stats[headers[index]] = cell.text.strip()
            player_data[player_name] = stats
    
    return player_data

# URL of the NBA player statistics
# year = 2023
# player_stats = fetch_nba_player_stats(year)
# print(player_stats)

# # Optionally, print out a few entries to check
# for name, stats in list(player_stats.items())[:5]:  # Print stats for the first 5 players
#     print(f"{name}: {stats}")

#抓取球員資料函數
def get_stats(year):
    url='https://www.basketball-reference.com/leagues/NBA_' + year +'_per_game.html'
    stats=pd.read_html(url)[0]
    stats['year']=year
    return stats