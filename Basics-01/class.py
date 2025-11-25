# class car:
#     def __init__(self, brand, model):
#         self.brand = brand
#         self.model = model

# cars = [
#     car("Koeniggsegg", "Regera"),
#     car("Ferarri", "LaFerarri"),
#     car("Buggati", "Chiron"),
# ]

# for car in cars:
#     print(f"Brand: {car.brand}, Model: {car.model}")

class car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

cars = []

n = int(input("How many cars do you want to enter:"))

for i in range(n):
    brand = input(f"Enter the brand name for car {i+1}:")
    model = input(f"Enter the model of the car for brand{i+1}:")
    cars.append(car(brand, model))

for car in cars:
    print(f"Brand: {car.brand}, Model: {car.model}")