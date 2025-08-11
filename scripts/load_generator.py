import requests
import random
import time


API_URL = "http://localhost:8000/predict"
# API_URL = "http://api:8000/predict"
DURATION_SECONDS = 30
SLEEP_BETWEEN = 0.2

def generate_random_payload():
    """Generate a random Iris measurement payload."""
    return {
        "sepal_length": round(random.uniform(4.5, 7.5), 1),
        "sepal_width": round(random.uniform(2.0, 4.5), 1),
        "petal_length": round(random.uniform(1.0, 6.5), 1),
        "petal_width": round(random.uniform(0.1, 2.5), 1)
    }

def main():
    start_time = time.time()
    count = 0
    print(f"Sending requests to {API_URL} for {DURATION_SECONDS} seconds...")

    while time.time() - start_time < DURATION_SECONDS:
        payload = generate_random_payload()
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                count += 1
            else:
                print(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        time.sleep(SLEEP_BETWEEN)

    print(f"Completed {count} requests in {DURATION_SECONDS} seconds.")

if __name__ == "__main__":
    main()
