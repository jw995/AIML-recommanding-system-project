# similar item model
# underneath it is to use a clustering model
# for simplicity, return all in the same cluster if rating is higher or equal to 3; return empty cluster otherwise

class SimilarItemModel():
	THRESHOLD = 3.0 # if ratings are below threshold, it will not be used

	def __init__(self, clusteringModel):
		# we use a trained clustering model (trained offline)
		self.clusteringModel = clusteringModel
		self.recs = []

	def train(self, itemFeature, rating):
		# itemFeature: the feature of the item in the Action
		# rating: the rating of the user to the item, also in the Action
		# only single record
		# each model learns one person's current interest
		itemFeature = itemFeature.values.reshape(1,-1)
		center, indices = self.clusteringModel.predict(itemFeature)

		# indices: the itemIds that are in the same cluster as the item we get
		# that is, the similar items

		if rating >= self.THRESHOLD:
			self.recs = indices[0] # the indices is a list of list, like: [[1,2,3,4,5]]
		else:
			self.recs = []

	def predict(self, itemFeature):
		# X should be item's category feature, only single record
		# return the similar items
		itemFeature = itemFeature.values.reshape(1,-1)
		center, indices = self.clusteringModel.predict(itemFeature)
		return indices[0]

	def provideRec(self):
		return self.recs


if __name__=="__main__":
	from DatabaseInterface import DatabaseInterface
	from Models.ClusteringModel import ClusteringModel
	db = DatabaseInterface("../DATA")
	db.startEngine()
	itemFeatureTable = db.extract(DatabaseInterface.ITEM_FEATURE_KEY).loc[:, "unknown":]

	model = ClusteringModel()
	model.train(itemFeatureTable)

	modelSI = SimilarItemModel(model)
	modelSI.train(itemFeatureTable.loc[1], 4)
	print modelSI.provideRec()
	modelSI.train(itemFeatureTable.loc[1], 2)
	print modelSI.provideRec()
