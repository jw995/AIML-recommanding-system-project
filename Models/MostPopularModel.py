# most popular model
# here it is a simple design: find the one with highest score with most of the users

class MostPopularModel():
	N_Freq_limit = 0.001 # at least 0.1% of users have rated it, we can start to consider if it is most popular

	def __init__(self):
		pass

	def train(self, history):
		# X must be a dataframe, with the second key as itemID, and third key as ratings
		itemID = list(history)[1]
		ratings = list(history)[2]

		# what if only an item only got rated by one user, and the rating is 5, are we confident it is most popular?
		nLimit = int(history.shape[0]*self.N_Freq_limit)
		itemRatingGrouped = history.groupby(itemID)
		itemRatingGroupedCount = itemRatingGrouped[ratings].count()
		# print itemRatingGrouped[ratings].mean()
		self.mostPopular = itemRatingGrouped[ratings].mean()[itemRatingGroupedCount>nLimit].sort_values(ascending=False)

	def predict(self,X):
		# X can only be a list of itemID's
		return [self.mostPopular.index.get_loc(x) for x in X]

	def provideRec(self):
		return self.mostPopular.index.tolist()



if __name__=="__main__":
	from DatabaseInterface import DatabaseInterface
	db = DatabaseInterface("DATA")
	db.startEngine()
	df = db.extract("history")
	print df.head()
	model = MostPopularModel()
	model.train(df)
	print model.mostPopular
	print model.predict([408])
	print model.provideRec()