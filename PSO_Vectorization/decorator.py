import functools
import time


def bench(func):
    
    @functools.wraps(func)
    def wrapper_brench(*args, **kwargs):
        start_time = time.perf_counter()         
        res = func(*args)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        return run_time,res
    
    return wrapper_brench