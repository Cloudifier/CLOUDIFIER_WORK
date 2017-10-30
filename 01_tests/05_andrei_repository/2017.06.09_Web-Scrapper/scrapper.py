from bs4 import BeautifulSoup as bsoup
import requests as rq
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model
import matplotlib.pyplot as plt

class LinearRegression():

	def __init__(self, df, test_size):
		self.df = df.copy()
		self.model = None
		self.test_size = test_size
		self.number = None

	def label_encoding(self):
		self.number = LabelEncoder()
		self.df['Tip'] = self.number.fit_transform(self.df['Tip'].astype(str))

	def fillna(self):
		self.df['Camere'].replace(["n/a", "5+"], ["0", "5"], inplace = True)
		self.df['Camere'] = pd.to_numeric(self.df['Camere'])

	def treat_outliers(self):
		self.df.drop(self.df[ self.df['Pret'] > 500000 ].index, inplace =  True)
		self.df.drop(self.df[ self.df['Tip'] == "Terenuri" ].index, inplace =  True)
		self.df = self.df.reset_index(drop=True)

	def preprocessing(self):
		self.treat_outliers()
		self.fillna()
		self.label_encoding()

	def train_model(self):

		print("Train model..................................................")
		train_df = self.df[:-self.test_size]

		predictors = ['Tip', 'Suprafata', 'Camere']

		x_train = train_df[predictors].values
		y_train = train_df['Pret'].values

		self.model = sklearn.linear_model.LinearRegression()
		self.model.fit(x_train, y_train)

	def make_prediction(self):

		print("Make prediction based on trained model.......................")
		test_df = self.df[-self.test_size:]

		predictors = ['Tip', 'Suprafata', 'Camere']
		x_test = test_df[predictors].values

		predictions = self.model.predict(x_test)

		return predictions

	def compare_with_real(self, predictions):

		self.df['Tip'] = self.number.inverse_transform(self.df['Tip'])

		reals = []
		mp2 = []
		for i in range(self.test_size):
			crt_idx = len(self.df) - self.test_size + i
			print("Real estate to predict:")
			print("Type: {} , Mp2: {} , Rooms: {} , Price: {}".format( \
				  self.df.ix[crt_idx][0], self.df.ix[crt_idx][1], \
				  self.df.ix[crt_idx][2], self.df.ix[crt_idx][3]))
			print("Predicted price: {}".format(predictions[i]))
			print()
			reals.append(df.ix[crt_idx][3])
			mp2.append(df.ix[crt_idx][1])

		plt.xlabel('Surface (MP2)')
		plt.title('Real values and predicted ones')
		plt.ylabel('Price (EUR)')
		plt.plot(mp2, reals, '.', mp2, predictions, '-')
		plt.show()

		self.label_encoding()


class HousingPriceCrawler():

	def __init__(self, url, ignore, num_pages):
		self.url = url
		self.ignore = ignore
		self.num_pages = num_pages
		self.property_types = []
		self.sizes = []
		self.rooms = []
		self.prices = []

	def crawl(self):
		r = rq.get(self.url)
		soup = bsoup(r.text,"lxml")

		for i in range(self.num_pages):

			page_url = self.url[:-5] + "_" + str(i + 2) + ".html"

			ads_details = soup.findAll("div", {"class": "info_details"})
			ads_per_page = 0
			for ad in ads_details:

				details = ad.findAll("span")
				ads_per_page += 1

				self.property_types.append(details[0].text)

				if len(details) == 1:
					self.sizes.append(0.0)
					self.rooms.append("n/a")
					continue

				mp2 = details[1].text[:-2].replace(',', '.');
				self.sizes.append(float(mp2))

				if len(details) == 2:
					self.rooms.append("n/a")
					continue

				self.rooms.append(details[2].text[:-len("camere") - 1])
				if self.rooms[-1].find("o") != -1:
					self.rooms[-1] = "1"

			print("Parsed page number {} with {} entries".format(
				i + 1, ads_per_page))
			ads_prices = soup.findAll("span", {"class": "price"})
			for ad in ads_prices:

				if any(x in ad.text for x in self.ignore):
					continue

				price = ad.text.replace('.', '');
				price = price.replace(',', '.');
				self.prices.append(float(price[:-4]))

			print("Last property from this page:")
			print("Type: {} , Mp2: {} , Rooms: {} , Price: {}".format( \
				  self.property_types[-1], self.sizes[-1], \
				  self.rooms[-1], self.prices[-1]))
			print()

			r = rq.get(page_url)
			soup = bsoup(r.text,"lxml")

	def get_dataframe(self):

		values = [('Tip', self.property_types), ('Suprafata', self.sizes), \
		  		  ('Camere', self.rooms), ('Pret', self.prices)]

		return pd.DataFrame.from_items(values)

if __name__ == "__main__":

	base_url = "https://homezz.ro/anunturi_in-bucuresti-if.html"
	web_crawler = HousingPriceCrawler(base_url, ["de la", "maxim"], 50)
	web_crawler.crawl()

	df = web_crawler.get_dataframe()

	linear_model = LinearRegression(df, 50)
	linear_model.preprocessing()
	linear_model.train_model()
	predictions = linear_model.make_prediction()

	linear_model.compare_with_real(predictions)






