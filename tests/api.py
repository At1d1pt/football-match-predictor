import requests

home = input("Enter home team: ").title()
away = input("Enter away team: ").title()
date = input("Enter match date in the form of YYYY-MM-DD (AFTER 2020-08-01): ")

d = {
    "home": home,
    "away": away,
    "date": date
}

r = requests.post("http://127.0.0.1:8000/predict", json=d).json()
print(r)