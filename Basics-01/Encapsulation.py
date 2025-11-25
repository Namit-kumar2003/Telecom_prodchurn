class car:
    def __init__(self, brand, model): #This is an example of encapsulation. That means making some attributes of this class
        self.__brand = brand          #private.
        self.model = model
    def get_brand(self):              #This is called getter, with this only now I can access the brand attribute
        return self.__brand
    

cars = car("Ferrari", "LaFerrari")
print(f"Brand: {cars.get_brand()}, Model: {cars.model}")


