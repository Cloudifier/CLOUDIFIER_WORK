class Laptop():

	def __init__(self, cpus = 0, capacity = 0, hasSSD = False):
		self.cpus = cpus
		self.capacity = capacity
		self.hasSSD = hasSSD

	def __str__(self):
		return "(cpus = {}, capacity = {}, hasSSD = {})".format(self.cpus, self.capacity, self.hasSSD)

	def __repr__(self):
		return self.__str__()

laptop1 = Laptop()
laptop2 = Laptop(2, 500, True)

def zipLongest(l1, l2):
	auxL1 = l1
	auxL2 = l2
	diff = len(l1) - len(l2)
	diff = abs(diff)

	if len(l1) > len(l2):
		[auxL2.append(0) for i in range(diff)]
	else:
		[auxL1.append(0) for i in range(diff)]

	return zip(auxL1, auxL2)


def isPalidrome(word):
	return word == str.join('', reversed(word))

def palindromeMap(words_list):
	return ["Yes" if isPalidrome(elem) else "No" for elem in words_list]

class Father():

	def __init__(self):
		print("Father constructor")

	def function(self):
		print("Father function")

class Child(Father):

	def __init__(self):
		print("Child constructor")

	def function(self):
		print("Child function")
		self.function()


child = Child()
child.function()
