class SliceTest:
    def __init__(self, a, b):
        self.a = a
        self.b = b

class TestClass:
    def __init__(self):
        self.a = [1, 2, 3]
        self.b = [4, 5, 6]

    def __getitem__(self, item):
        return SliceTest(self.a[item], self.b[item])
    def __setitem__(self, key, value):
        self.a[key] = value.a
        self.b[key] = value.b


t = TestClass()
slice_test = t[1]
slice_test.a = 15
t[1] = slice_test
print('w')