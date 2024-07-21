import os
import glob
import json
import socket
from urllib.parse import urlparse
import re
import ipaddress
import requests
import time
import csv

def get_ip_from_url(url):
    # Parse the URL to extract the domain
    parsed_url = urlparse(url)
    domain = parsed_url.hostname

    # Get the IP address of the domain
    try:
        ip_address = socket.gethostbyname(domain)
        return ip_address
    except socket.gaierror as e:
        return f"Error: {e}"

def is_ip_address(string):
    try:
        ipaddress.ip_address(string)
        return True
    except ValueError:
        return False

def check_malicious_ip(ip_address):
    url = f'http://isc.sans.edu/api/ip/{ip_address}?json'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        #print(data)

        if data:
            #print(f"IP Address: {ip_address}")
            count = 0
            attacks = 0
            threatfeeds = {}

            if data['ip']:
                if data['ip'].get('count'):
                    count = data['ip'].get('count')

                if data['ip'].get('attacks'):
                    attacks = data['ip'].get('attacks')

                if data['ip'].get('threatfeeds'):
                    threatfeeds = data['ip'].get('threatfeeds')

                #print(f"count: {data['ip'].get('count')}")
                #print(f"attacks: {data['ip'].get('attacks')}")
                #print(f"threatfeeds: {data['ip'].get('threatfeeds')}")

            if count > 200 or attacks > 20 or threatfeeds:
                return True
            else:
                return False
        else:
            print(f"No data found for IP: {ip_address}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Error checking IP: {e}")


folder_path = r'C:\Users\atuwh\Documents\YL\Gatech\INTA6450\NowSecure\Project_DAS\preprocess\MaliciousListMatching\NowSecureJSONData\Group1to7'
#folder_path = r'C:\Users\atuwh\Documents\YL\Gatech\INTA6450\NowSecure\test\Group 1'
json_files = glob.glob(os.path.join(folder_path, '*.json'))
result = {}

for json_file in json_files:
    try:
        print(json_file)
        with open(json_file, 'r',  encoding='utf-8') as file:
            try:
                json_data = json.load(file)

                if len(json_data) > 0:
                    for item in json_data:
                        try:

                            if len(item['title']) > 0 and item['assessment']:
                                title = item['title']
                                print(item['title'])
                                urls = []
                                malicious_url_count = 0
                                if len(item['assessment']['analysis']) > 0:
                                    if len(item['assessment']['analysis']['task']) > 0:
                                        if len(item['assessment']['analysis']['task']['static']) > 0:
                                            if len(item['assessment']['analysis']['task']['static']['result']) > 0:
                                                if len(item['assessment']['analysis']['task']['static']['result'][
                                                           'urls_check']) > 0:
                                                    if len(item['assessment']['analysis']['task']['static']['result'][
                                                               'urls_check']['urls']) > 0:
                                                        urls = \
                                                        item['assessment']['analysis']['task']['static']['result'][
                                                            'urls_check']['urls']
                                                        if len(urls) > 0:
                                                            for url in urls:

                                                                try:
                                                                    if len(url) > 0:
                                                                        # print(url['url'])
                                                                        temp = get_ip_from_url(url['url'])
                                                                        if is_ip_address(temp):
                                                                            print(f"IP address of {url['url']}: {temp}")
                                                                            if check_malicious_ip(temp):
                                                                                malicious_url_count = malicious_url_count + 1
                                                                            time.sleep(1.5)
                                                                except Exception as e:
                                                                    print(f"Error processing URL {url}: {e}")
                                                                    continue

                                print(f"Malicious URL Count: {malicious_url_count}")
                                result[title] = malicious_url_count

                        except KeyError as e:
                            print(f"Missing expected key in item: {e}")
                            continue
                        except Exception as e:
                            print(f"Error processing item: {e}")
                            continue

            except json.JSONDecodeError as e:
                print(f"JSON decode error in file {json_file}: {e}")
                continue

    except Exception as e:
        print(f"Error processing file {json_file}: {e}")
        continue

print(f"Result: {result}")


csv_file = r'C:\Users\atuwh\Documents\YL\Gatech\INTA6450\NowSecure\Project_DAS\preprocess\MaliciousListMatching\malicious_urls_count.csv'

try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['Title', 'Malicious URL Count'])

        # Write the data
        for title, count in result.items():
            csvwriter.writerow([title, count])

    print(f"Results successfully written to {csv_file}")
except Exception as e:
    print(f"Error writing to CSV file: {e}")






