import time
import progressbar

bar = progressbar.ProgressBar(redirect_stdout=True)

custom_widgets=[
		  progressbar.RotatingMarker(),
    '[',  progressbar.Percentage(), ']',
          progressbar.RotatingMarker(),
          progressbar.Bar('=', '|', '|'),
    ' (', progressbar.AdaptiveETA(), ') ',
    	  progressbar.SimpleProgress(),
]

bar = progressbar.ProgressBar(widgets = custom_widgets, redirect_stdout=True)

for i in bar(range(200)):
    print('Some text {}'.format(i))
    time.sleep(0.1)
