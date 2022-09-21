Year = int

class Days:
    def __init__(self, value):
        self.value = value

    def days(self):
        return self.value

    def hours(self):
        return int(self.value * 24)

class Hours:
    def __init__(self, value):
        self.value = value
    
    def days(self):
        return self.value // 24

    def hours(self):
        return self.value