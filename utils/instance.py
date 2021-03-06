class Instance(object):
    def __init__(self, query=None, doc=None, match=None, category=None):
        self.query = query
        self.doc = doc
        self.match = match
        self.category = category

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)



