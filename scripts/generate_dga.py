#!/usr/bin/env python3
import hashlib
import random
import string
from datetime import datetime, timedelta

def cryptolocker_dga(date, count=1000):
    domains = []
    for i in range(count):
        seed = f"{date.strftime('%Y%m%d')}{i}"
        h = hashlib.sha256(seed.encode()).hexdigest()
        length = 12 + (int(h[:2], 16) % 8)
        domain = ""
        for j in range(length):
            idx = (j * 2) % 60
            domain += chr(ord('a') + (int(h[idx:idx+2], 16) % 26))
        tld = ['.com', '.net', '.org', '.biz', '.info'][int(h[-2:], 16) % 5]
        domains.append(domain + tld)
    return domains

def necurs_dga(date, count=1000):
    domains = []
    for i in range(count):
        seed = f"necurs{date.year}{date.month}{i}"
        h = hashlib.sha256(seed.encode()).hexdigest()
        length = 15 + (int(h[:2], 16) % 10)
        domain = ""
        for j in range(length):
            idx = (j * 2) % 60
            domain += chr(ord('a') + (int(h[idx:idx+2], 16) % 26))
        tld = ['.com', '.net', '.org', '.top', '.xyz'][int(h[-2:], 16) % 5]
        domains.append(domain + tld)
    return domains

def random_dga(count=5000):
    domains = []
    tlds = ['.com', '.net', '.org', '.xyz', '.top', '.info', '.biz', '.cc', '.ru']
    for _ in range(count):
        length = random.randint(10, 25)
        chars = string.ascii_lowercase + string.digits
        domain = ''.join(random.choice(chars) for _ in range(length))
        domains.append(domain + random.choice(tlds))
    return domains

def conficker_dga(date, count=500):
    domains = []
    for i in range(count):
        seed = date.toordinal() + i
        random.seed(seed)
        length = random.randint(8, 12)
        domain = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        tld = random.choice(['.com', '.net', '.org', '.info', '.biz'])
        domains.append(domain + tld)
    return domains

def main():
    all_dga = []
    
    today = datetime.now()
    for days_back in range(30):
        date = today - timedelta(days=days_back)
        all_dga.extend(cryptolocker_dga(date, 500))
        all_dga.extend(necurs_dga(date, 500))
        all_dga.extend(conficker_dga(date, 300))
    
    all_dga.extend(random_dga(20000))
    
    all_dga = list(set(all_dga))
    random.shuffle(all_dga)
    
    output_path = "data/dga/dga_domains.txt"
    with open(output_path, 'w') as f:
        for domain in all_dga:
            f.write(domain + '\n')
    
    print(f"Generated {len(all_dga)} DGA domains")
    print(f"Saved to {output_path}")
    print(f"Sample:")
    for d in all_dga[:10]:
        print(f"  {d}")

if __name__ == "__main__":
    main()
