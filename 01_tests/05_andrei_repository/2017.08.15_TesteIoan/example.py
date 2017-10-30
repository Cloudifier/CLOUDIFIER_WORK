class MyClass(object):
	def __init__(self):
		self.a = 1
		self.b = 2
		self.c = None

	#def __str__(self):
	#	return "{} {} {}".format(self.a, self.b, self.c)

	def __repr__(self):
		return "{} {} {}".format(self.a, self.b, self.c)
		#return self.__str__()


l = [MyClass() for i in range(3)]
print(l)
