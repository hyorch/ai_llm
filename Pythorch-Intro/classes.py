class animal:
    rango = "animal"
    def __init__(self, name, age, saludo):
        self.name = name
        self.age = age   
        self.saludo = saludo     
        

    def speak(self):
        print(self.saludo,":",self.name)

class dog(animal): 
    def ladrar(self):
        return super().speak()
    

class cat(animal):
    def maullar(self):
        return super().speak()
    

dog = dog("Buddy", 3, "Guau")
dog.ladrar()

cat = cat("Rony", 5, "Miau")
cat.maullar()


class frog(animal):
    def croar(self):
        return super().speak()

frog = frog("Pepe", 1, "Croac")
frog.croar()



