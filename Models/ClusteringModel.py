from sklearn.cluster import KMeans

# clustering model is to group items with similar features
# it is used for online recommendation

class ClusteringModel():
	def __init__(self, n_cluster=10):
		self.model = KMeans(n_cluster, random_state=12345) # set random state for reproducible
		self.groups = {} # keyed by cluster index and values are itemId's
		self.trained = False

	def train(self, itemFeatures):
		self.indices = itemFeatures.index # the itemIds
		self.model.fit(itemFeatures)
		self.labels = self.model.labels_
		# the label given for each data point
		# the label indicates which cluster it belongs to
		# for example, if we have four data, 1,2,3,4
		# dataId, clusterId
		# 1			1
		# 2			1
		# 3			2
		# 4			2

		# and we want {1:[1,2],2:[3,4]}, called self.groups
		for k, v in zip(self.labels, itemFeatures.index.tolist()):
			self.groups.setdefault(k,[]).append(v)
		self.trained = True

	def predict(self, itemFeatures):
		centers = self.model.predict(itemFeatures)

		# based on the predicted centers, find the corresponding cluster members
		return centers, [self.groups[c] for c in centers]


if __name__=="__main__":
	from DatabaseInterface import DatabaseInterface
	db = DatabaseInterface("../DATA")
	db.startEngine()
	itemFeatureTable = db.extract(DatabaseInterface.ITEM_FEATURE_KEY).loc[:, "unknown":]

	model = ClusteringModel()
	model.train(itemFeatureTable)

	print model.predict(itemFeatureTable.loc[1].values.reshape(1,-1))
	print itemFeatureTable.loc[[1,422]]
	print model.labels[:20]