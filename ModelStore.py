# model store
# keep all the models
# responsible to send the models to RecEngine

from Models.ClusteringModel import ClusteringModel
from Models.SimilarItemModel import SimilarItemModel
from Models.CFmodel import CFmodel
from Models.MostPopularModel import MostPopularModel
from Models.KNNmodel import KNNmodel


class ModelStore(object):

	CF_MODEL_KEY = "cf_model_key" # collaborative filtering
	KNN_MODEL_KEY = "knn_model_key" # K nearest neighbor most popular model
	MP_MODEL_KEY = "mp_model_key" # most popular
	SI_MODEL_KEY = "si_model_key" # similar item
	CL_MODEL_KEY = "cl_model_key" # clustering model, used for similarity item model

	def __init__(self):
		self.persistModels = {self.KNN_MODEL_KEY: KNNmodel(),
		                      self.MP_MODEL_KEY: MostPopularModel(),
		                      self.CL_MODEL_KEY: ClusteringModel(),
		                      self.CF_MODEL_KEY: CFmodel()}

		# similarity model is used for each user
		# online recommendation, trained by online learner
		self.transientModels = {self.SI_MODEL_KEY: {}}

	def setModel(self, model, key, memberId = None):
		if memberId is None:
			self.persistModels[key] = model
		else:
			self.transientModels[key][memberId] = model

	def getModel(self, key, memberId = None):
		#send out the object of models to learning system
		if memberId is None:
			return self.persistModels[key]
		else:
			transientModels = self.transientModels[key]
			if memberId in transientModels:
				return transientModels[memberId]
			else:
				# it means it is the first time we build the online model for this particular user
				assert self.persistModels[self.CL_MODEL_KEY].trained
				# since the online model we used is depending on the clustering model trained offline
				# this assert is to make sure the clustering model is already trained.

				# in this case, we create a new model for this user
				return SimilarItemModel(self.persistModels[self.CL_MODEL_KEY])

	def cleanOnlineModel(self):
		self.transientModels = {self.SI_MODEL_KEY: {}}

