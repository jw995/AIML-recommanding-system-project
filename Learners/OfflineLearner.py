# Offline Learner
# Read in user history data, make models

from ModelStore import ModelStore
from DatabaseInterface import DatabaseInterface
import logging
import numpy as np

class OfflineLearner(object):
	logging.basicConfig(level=logging.INFO)

	def __init__(self, database, modelStore):
		self.database = database
		self.modelStore = modelStore
		self.log = logging.getLogger(__name__)
		self.modelRegistry = [(ModelStore.KNN_MODEL_KEY, "k nearest neighbor most popular model"),
								(ModelStore.MP_MODEL_KEY, "most popular item model"),
								(ModelStore.CL_MODEL_KEY, "item feature clustering model"),
								(ModelStore.CF_MODEL_KEY, "collaborative filtering model")]

	def trainModel(self):
		self.log.info("Start offline training...")
		self.log.info("creating training data...")
		# now extract data
		# K nearest neighbor is to provide similar users,
		# 	trained on user feature,
		# 	given a user feature, predict the nearest users
		#	recommend based nearest neighbor's result (use user item rating matrix)
		# most popular item
		#	trained on user item rating matrix
		#	predict most popular items
		#	recommend most popular items
		# item feature clustering model (only used for online model and CN)
		# 	trained on item feature
		#	given item features, predict groups
		# collaborative filtering model
		#	trained on user item rating matrix
		#	given a userId or itemId, give the full prediction
		#	recommend high predicted ratings
		historyRating = self.database.extract(DatabaseInterface.HISTORY_KEY)
		itemFeatureTable = self.database.extract(DatabaseInterface.ITEM_FEATURE_KEY).loc[:, "unknown":]
		userFeatureTable = self.database.extract(DatabaseInterface.USER_FEATURE_KEY).loc[:, "age":]
		ratingsMat = self.transformToMat(historyRating)

		# update model and push back
		# for offline model, here we only implement fully retrain, but one should always have a backup
		# here we directly update the current model
		self.log.info("loading models...")
		for record in self.modelRegistry:
			model = self.modelStore.getModel(record[0])
			self.log.info("training %s" %record[1])
			if record[0] == ModelStore.KNN_MODEL_KEY:
				model.train(userFeatureTable, ratingsMat)
			elif record[0] == ModelStore.MP_MODEL_KEY:
				model.train(historyRating)
			elif record[0] == ModelStore.CL_MODEL_KEY:
				model.train(itemFeatureTable)
			elif record[0] == ModelStore.CF_MODEL_KEY:
				model.train(ratingsMat, itemFeatureTable)
			else:
				raise Exception("model registry may be broken")

			self.log.info("updating %s", record[1])
			self.pushModel(model, record[0])


	def pushModel(self, model, key):
		self.modelStore.setModel(model, key)

	@staticmethod
	def transformToMat(historyRating):
		n_users = historyRating.user_id.max()
		n_items = historyRating.item_id.max()
		ratingsMat = np.zeros([n_users, n_items])
		for r in historyRating.itertuples():
			ratingsMat[r[1]-1,r[2]-1] = r[3]
		return ratingsMat


if __name__=="__main__":
	db = DatabaseInterface("DATA")
	db.startEngine()
	modelStore = ModelStore()
	learner = OfflineLearner(db, modelStore)
	learner.trainModel()