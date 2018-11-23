class auto:

    def __init__(self,name):

        self.name = name
        self.pocet_kol = 4
        print('nove auto je na svete')


    def jed(self):
        print('jedu rovne a jsem auto'+self.name)



mercedes = auto('mercedes')

mercedes.jed()
