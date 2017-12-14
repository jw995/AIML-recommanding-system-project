# KNN model
# for new user, using user feature, find the k nearest neighbor, using their ratings for the recommendation

import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNmodel():
	def __init__(self):
		self.knnModel = None

	def train(self, userFeatureTable, ratingsMat):
		userFeatureTable.loc[:,"age"] = userFeatureTable.loc[:,"age"]/10.
		# ad hoc fix, make sure feature's range is similar
		self.knnModel = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(userFeatureTable)

		# ratingMat is the rating matrix
		self.ratingsMat = ratingsMat
		self.userFeatureTable = userFeatureTable
		self.userIds = self.userFeatureTable.index # the actual order seen by the knnmodel

	def predict(self, userFeature):
		distances, indices = self.knnModel.kneighbors(userFeature)

		# indices are the nearest neighbors' index in the matrix, which is different from userId.
		return self.userIds[indices[0]]

	def provideRec(self, userId):
		#data is a tuple of (user feature, item feature)
		userIds = self.predict(self.userFeatureTable.loc[userId].as_matrix().reshape(1,-1))
		# remove himself as a nearest neighbor
		userIds = np.array(list(set(userIds) - set([userId])))

		# for all nearest neighbors, compute the the average score, sorted from large to small
		# then report the item ids
		return self.ratingsMat[userIds-1].mean(axis = 0).argsort()[::-1]+1



if __name__=="__main__":
	from DatabaseInterface import DatabaseInterface
	from Learners.OfflineLearner import OfflineLearner
	db = DatabaseInterface("../DATA")
	db.startEngine()
	history = db.extract("history")
	userFeatureTable = db.extract(DatabaseInterface.USER_FEATURE_KEY).loc[:, "age":]
	ratingsMat = OfflineLearner.transformToMat(history)

	model = KNNmodel()
	model.train(userFeatureTable, ratingsMat)
	print model.provideRec(97)[:20]
	print ratingsMat[96,model.provideRec(97)-1][:20]
