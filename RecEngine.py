# Recommendation Engine

from ModelStore import ModelStore
import logging

class RecEngine(object):
	logging.basicConfig(level=logging.INFO)

	def __init__(self, userAnalyzer, modelStore, userActivityTable):
		self.userAnalyzer = userAnalyzer
		self.modelStore = modelStore
		self.userActivityTable = userActivityTable
		self._cacheMostPopular()
		# to pre-compute the most popular items, because this recommendation is independent from users
		self.log = logging.getLogger(__name__)

	def resetCache(self):
		self._cacheMostPopular()

	def _cacheMostPopular(self):
		self.mostPopularList = self.modelStore.getModel(ModelStore.MP_MODEL_KEY).provideRec()

	def provideRecommendation(self, request):
		recommendations = {}
		# dictionary, dict1 = {"key":"value"}, when I try to get the value, I can use dict1["key"]
		# construct recommendation content, which is implemented as a dictionary
		# three sections will be used: popular, online, offline

		recommendations["popular"] = self.mostPopularList
		requestAnalyzed = self.userAnalyzer.analyze(request, self.userActivityTable)

		# online recommendation
		onlineRecs = self.modelStore.getModel(ModelStore.SI_MODEL_KEY, request.userId).provideRec()

		self.log.info("user type: %s" %requestAnalyzed[0])

		# now we start to construct our recommendation data
		if len(onlineRecs)>0:
			recommendations["online"] = onlineRecs # a lit of ids

		if requestAnalyzed[0] == "new":
			# for new user, we use KNN model for offline model recommendation
			recommendations["offline"] = self.modelStore.getModel(ModelStore.KNN_MODEL_KEY)\
				.provideRec(requestAnalyzed[2].userId)
		elif requestAnalyzed[0] == "old":
			# for new user, we use CF model for offline model recommendation
			recommendations["offline"] = self.modelStore.getModel(ModelStore.CF_MODEL_KEY)\
				.provideRec(requestAnalyzed[2].userId)

		return requestAnalyzed[1], recommendations


