from tqdm import tqdm, trange
from time import sleep
from logger import Logger
import inspect


logger = Logger(show = True)
old_print = print
inspect.builtins.print = tqdm.write


t = trange(500, desc='Bar desc', leave=True)
for i in range(1, 501):
	logger.log("Ceva {}".format(i))
	t.set_description("Description no {}/{}".format(i, 500))
	t.refresh()
	t.update(1)
	sleep(0.1)

