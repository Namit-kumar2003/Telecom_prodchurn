# import time

# def timer(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = func(*args, **kwargs)
#         end = time.time()
#         print(f"{func.__name__} ran in {end-start} time")
#         return result
#     return wrapper

# @timer
# def example_function(n):
#     time.sleep(n)

# example_function(2)

def debug(func):
    def wrapper(*args, **kwargs):
        args_Value = ', '.join(str(arg) for arg in args)
        kwargs_value = ', '.join(f"{k}: {v}" for k, v in kwargs.items())
        print(f"Calling {func.__name__} with args {args_Value} and keyword arguments {kwargs_value}")
        return func(*args, **kwargs)
    return wrapper

@debug
def Greet(name, greeting):
    print(f"{greeting}, {name}")

Greet("Dharampal", greeting = "Ram Ram")