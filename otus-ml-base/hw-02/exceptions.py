"""
Объявите следующие исключения:
- LowFuelError
- NotEnoughFuel
- CargoOverload
"""

class LowFuelError(Exception):
    pass

class NotEnoughFuel(Exception):
    pass

class CargoOverload(Exception):
    pass

if __name__ == "__main__":
        try:
            raise LowFuelError
        except LowFuelError:
            print('You have low fuel!')

        try:
            raise NotEnoughFuel
        except NotEnoughFuel:
            print('You have not enough fuel!')

        try:
            raise CargoOverload
        except CargoOverload:
            print('The vehicle is overloaded!')
