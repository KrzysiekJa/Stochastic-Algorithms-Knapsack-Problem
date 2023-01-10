import functools
import time
def bench(func):
    @functools.wraps(func)
    def wrapper_brench(*args, **kwargs):

        #taking the start time
        start_time = time.perf_counter()         
        res = func(*args)
        print("Result : "+str(res))
        #taking the end time
        end_time = time.perf_counter()
        run_time = end_time - start_time
        return run_time
    return wrapper_brench