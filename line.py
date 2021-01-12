import math


class Line():
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def getM(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def getQ(self):
        return -(self.getM()*self.x1 + self.y1)

    def getLenght(self):
        return math.sqrt((self.x2 - self.x1)*(self.x2 - self.x1) + (self.y2 - self.y1)*(self.y2 - self.y1))

    def getEquation(self):
        a = self.y1 - self.y2
        b = self.x2 - self.x1
        c = (self.x1 - self.x2)*self.y1 + (self.y2 - self.y1)*self.x1
        return (a, b, c)

    def isEqual(self,  line, linesDistance):
        product = (self.x2 - self.x1)*(line.x2 - line.x1) + \
            (self.y2 - self.y1)*(line.y2 - line.y1)

        if (abs(product / (self.getLenght() * line.getLenght())) < math.cos(math.pi / 60)):
            return False

        a1, b1, c1 = self.getEquation()
        a2, b2, c2 = line.getEquation()

        dist = math.inf

        if(a1/b1 == math.nan or abs(a1/b1) > 1):
            dist = abs(c1/a1 - c2/a2) / math.sqrt(1 + abs((b1/a1) * (b2/a2)))
        else:
            dist = abs(c1/b1 - c2/b2) / math.sqrt(1 + abs((a1/b1) * (a2/b2)))

        if (dist > linesDistance):
            return False

        return True

    def getIntersectionPoint(self, line):
        a1, b1, c1 = self.getEquation()
        a2, b2, c2 = line.getEquation()

        if (a1*b2 - a2*b1 == 0):
            return (-1, -1)

        y = round((c1*a2 - c2*a1) / (a1*b2 - a2*b1))
        x = round((b1*c2 - b2*c1) / (a1*b2 - a2*b1))

        return (x, y)
