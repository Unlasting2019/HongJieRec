import time
from colorama import Fore, Style

def log(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(Fore.RED+f"{kwargs['f_name']}",end='')
        ret = func(*args, **kwargs)
        end = time.time()
        print(f'\tcost:{round(end-start, 4)}s')
        Style.RESET_ALL
        return ret
    return wrapper
