import _random
def turn():
    sixes = 0
    while True:
        roll = _random.randint(1, 6)
        print(f"Rolled: {roll}")
        if roll != 6: 
            break
        sixes += 1
        if sixes == 3:
            print("Three 6s! Turn forfeited.")
            break
        print("Bonus roll!")
        turn()