import json


class readData:
    def __init__(self):
        self.data = None

    def readtxt(self, name):
        with open(f'data\{name}', 'r') as f:
            content = f.read()

        return content

    def readjson(self, names = ["massage.txt","hotels.txt", "restaurants.txt" ]):
        data = []

        for name in names:
            content = self.readtxt(name)
            data.append(json.loads(content))

        return data

     