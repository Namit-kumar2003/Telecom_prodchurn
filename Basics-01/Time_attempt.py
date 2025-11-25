import time

wait_time = 1
attempt = 0
max_retries = 5

while attempt < max_retries:
    print("Attempt", attempt+1, "Wait time is:", wait_time)
    time.sleep(wait_time)
    wait_time *= 2
    attempt += 1