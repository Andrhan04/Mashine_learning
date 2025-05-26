import json
from datetime import date
from datetime import datetime
today = date. today()


def write_exeption(Function : str, exept : str):
    Log = f"log\\logs_{today}.txt"
    time = datetime.now().time()
    data : dict = {"Function":Function, "Exept" : exept,  'time' : str(time)}
    with open(Log, "a+", encoding='utf-8') as write_file:
        json.dump(data, write_file, ensure_ascii=False)
        write_file.write('\n')
        
def write_log(Function : str):
    Log = f"log\\logs_{today}.txt"
    time = datetime.now().time()
    data : dict = {"Function":Function, "Exept" : "Good",  'time' : str(time)}
    with open(Log, "a+", encoding='utf-8') as write_file:
        json.dump(data, write_file, ensure_ascii=False)
        write_file.write('\n')
        
def write_my_message(Function : str, message : str):
    Log = f"log\\logs_{today}.txt"
    time = datetime.now().time()
    data : dict = {"Function":Function, "message" : message,  'time' : str(time)}
    with open(Log, "a+", encoding='utf-8') as write_file:
        json.dump(data, write_file, ensure_ascii=False)
        write_file.write('\n')