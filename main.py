from extract_coin_images import detect_coins

if __name__ == '__main__':
    import time

    start_time = time.time()
    detect_coins()
    # detect_coins(start="k1_h2_s7")
    seconds = (time.time() - start_time)
    print(f"Time needed to detect coins: {round(seconds,2)} s")

    # start_time = time.time()
    # get_coin_info()
    # seconds = (time.time() - start_time)
    # print(f"Time needed to read text and gather info on coins: {round(seconds,2)} s")
