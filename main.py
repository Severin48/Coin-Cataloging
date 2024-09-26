from detect_coins import detect_coins

if __name__ == '__main__':
    import time
    start_time = time.time()
    detect_coins()
    seconds = (time.time() - start_time)
    print(f"Time neededto detect coins: {round(seconds,2)} s")
