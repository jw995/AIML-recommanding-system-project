# A simulation framework
import logging

from DatabaseInterface import DatabaseInterface
from RecEngine import RecEngine
from Ranker import Ranker
from Learners.OfflineLearner import OfflineLearner
from Learners.OnlineLearner import OnlineLearner
from UserAnalyzer import UserAnalyzer
from ModelStore import ModelStore


class WebServer(object):
	logging.basicConfig(level=logging.INFO)

	def __init__(self, configMap):
		self.db = DatabaseInterface(configMap['data_dir'])
		# numberToServe: the number of items finally served to the users
		self.numberToServe = configMap['numberToServe']
		self.log = logging.getLogger(__name__)

	def start(self):
		# each object here simulates the API calls through network
		# passing an object A to the constructor of B means A will communication to B
		self.db.startEngine()
		self.ranker = Ranker(self.numberToServe, self.db)
		self.userAnalyzer = UserAnalyzer()
		self.modelStore = ModelStore()
		self.offlineLearner = OfflineLearner(self.db, self.modelStore)
		self.onlineLearner = OnlineLearner(self.db, self.modelStore)
		self.offlineLearner.trainModel()
		# when we start the webserver, we should let offline learner to train the models,
		# such that, after the start(), we can start to give recommendation
		self.recEngine = RecEngine(self.userAnalyzer, self.modelStore, self.db.extract(DatabaseInterface.USER_ACTIVITY_KEY))


	def getAction(self, action):
		assert(isinstance(action, Action))
		# taking the action from users
		self.onlineLearner.trainModel(action)
		# analyze action type, and save the registered user's action
		actionType = self.userAnalyzer.analyzeAction(action)
		if actionType == "registered":
			self.log.info("Recording action %s" %action)
			self.db.putAction(action)

	def provideRecommendation(self, request):
		# return the ID's for the recommended items
		assert(isinstance(request, Request))
		# provide recommendations to user
		self.log.info("responding to request: %s" %request)
		recommendations = self.recEngine.provideRecommendation(request)
		recsReranked = self.ranker.rerank(recommendations)
		return recsReranked # a list of item ids

	def renderRecommendation(self, request):
		assert(isinstance(request, Request))
		recsReranked = self.provideRecommendation(request)
		# for the purpose of testing, we sort the index, output item names
		# output is ordered by the id value
		return self.db.extract(DatabaseInterface.INVENTORY_KEY).loc[recsReranked].sort_index()

	def increment(self):
		self.log.info("incrementing the system, update the models")
		# increment the whole system by one day, trigger offline training
		self.offlineLearner.trainModel()
		self.modelStore.cleanOnlineModel()
		self.recEngine.resetCache()

	# for demo purpose, given an itemId, return the item name
	def getFromInventory(self, itemId):
		return self.db.extract(DatabaseInterface.INVENTORY_KEY).loc[itemId]

# simulate a web request
class Request(object):
	def __init__(self, userId):
		self.userId = userId

	def __str__(self):
		return "request for user: "+str(self.userId)

# simulate a tracking event or a user's rating
class Action(object):
	def __init__(self, userId, itemId,rating):
		self.userId = userId
		self.itemId = itemId
		self.rating = rating

	def __str__(self):
		return "user: %s, item: %s, rating %s" %(self.userId, self.itemId, self.rating)