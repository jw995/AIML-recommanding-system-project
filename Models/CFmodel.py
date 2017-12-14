# Collaborative filtering model
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging


class CFmodel():
	RARECASE_THRESHOLD = 5
	logging.basicConfig(level=logging.INFO)

	def __init__(self):
		self.knnModel = NearestNeighbors(n_neighbors=15)
		self.log = logging.getLogger(__name__)

	def _CFSVD(self, ratingsMat):
		user_ratings_mean = np.mean(ratingsMat, axis = 1) # mean over user ratings
		R_demeaned = ratingsMat - user_ratings_mean.reshape(-1, 1)
		from scipy.sparse.linalg import svds
		U, sigma, Vt = svds(R_demeaned, k = 10)
		sigma = np.diag(sigma)
		self.all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)


	def train(self, ratingsMat, itemFeatureTable):
		# the logic:
		# using content-based modeling for rare items, predict some ratings
		# using the ratings matrix filled with the predicted ratings from the content-based model to do matrix factorization
		# itemFeatureTable is used for content-based model, which will predict for those items with few ratings
		# SVD will be used for collaborative filtering after the rare items have enough ratings
		indices = itemFeatureTable.index
		self.knnModel.fit(itemFeatureTable)
		# print ratingsMat.shape
		assert(ratingsMat.shape[1] == itemFeatureTable.index.max())
		rareCases = np.where((ratingsMat>0).sum(axis=0) < self.RARECASE_THRESHOLD)[0]
		# if an item has less than 5 ratings, it is considered as a rare case
		# it is the 0-based matrix indices
		self.log.info("Number of rare cases: %s" %rareCases.shape[0])

		fillCount = 0
		ratingsMatFinal = ratingsMat.copy()
		for case in rareCases:
			if case+1 in itemFeatureTable.index:
				features  = itemFeatureTable.loc[case+1]
				neighbors = self.knnModel.kneighbors(features.values.reshape(1, -1), return_distance=False)[0]
				neighborPos = indices[neighbors]-1
				# compute the number of ratings got by the neighbors from each user
				target_count = (ratingsMat[:,neighborPos] > 0).sum(axis=1)
				# compute the predicted ratings generated from the content-based model
				target_ratings = ratingsMat[:,neighborPos].sum(axis=1).astype(float)/target_count
				#nonzero mean

				for i in range(ratingsMat.shape[0]):
					if ratingsMat[i, case] == 0 and target_count[i]>10:
						# if the rating is missing and in its neighbors, more than 10 ratings are available
						if target_ratings[i]!=0:
							ratingsMatFinal[i,case] = target_ratings[i]
							fillCount += 1

		# now we have the filled matrix for matrix factorization
		self.log.info("Number of ratings added by content-based model: %s" %fillCount)

		self._CFSVD(ratingsMatFinal)



	def predict(self, userId):
		return self.all_user_predicted_ratings[userId-1]

	def provideRec(self, userId):
		# data is a tuple of (user feature, item feature)
		# compute the the average score, sorted from large to small, then report the item ids
		return self.all_user_predicted_ratings[userId-1].argsort()[::-1]+1

if __name__=="__main__":
	from DatabaseInterface import DatabaseInterface
	from Learners.OfflineLearner import OfflineLearner
	db = DatabaseInterface("../DATA")
	db.startEngine()
	history = db.extract("history")
	itemFeatureTable = db.extract(DatabaseInterface.ITEM_FEATURE_KEY).loc[:, "unknown":]
	ratingsMat = OfflineLearner.transformToMat(history)

	model = CFmodel()
	model.train(ratingsMat, itemFeatureTable)

	recs = model.provideRec(1)
	print recs
	print ratingsMat[0,recs-1]