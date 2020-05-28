class Person:
    kind = "human"
    #constructor. Default constructor defined by setting equal
    def __init__(self, name=None, height=None, age=None, sex=None, nickname = None):
        self.name = name
        self.height = height
        self.age = age
        self.sex = sex
        self.nickname = nickname
        self.trial = age + 5

    #When we use a class method, we need to have the self in there to reference instances within the class
    def create_nickname(self,name):
        self.nickname = name
        self.kind = "female"


Person_trial = Person("Gregory", 72, 21, "male")
Person_trial.create_nickname("Greg")
Person_trial.kind

print("Hello World")

2+2
Person_trial.trial
