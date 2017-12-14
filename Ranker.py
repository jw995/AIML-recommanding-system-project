# Ranker

import logging
import numpy as np

# rank the items from each recommendation module
# highly influenced by business strategy and varies from system to system
from DatabaseInterface import DatabaseInterface


class Ranker(object):
	logging.basicConfig(level=logging.INFO)
	def __init__(self, numberToServe, database):
		self.numberToServe = numberToServe
		self.userHistoryDB = database.extract(DatabaseInterface.HISTORY_KEY) # who rated what
		self.log = logging.getLogger(__name__)

	def _getUsedItems(self, userId):
		# return a python set of all the movies that have been seen
		if userId == -1 :
			return set([])
		else:
			return set(self.userHistoryDB[self.userHistoryDB.loc[:,"user_id"]==userId].loc[:,"item_id"])

	def rerank(self,recommendationsTuple):
		# recommendationTupe is a tuple of (userId, recommendations)
		# recommendations is a dictionary of lists {RecType: Items}, RecType can be "online", "offline", "popular"
		# return the ranked recommendation
		# here is the strategy:
		# if the userId is -1, it means it is from anonymous user.
		# else remove the watched item and

		userId = recommendationsTuple[0]
		recommendations = recommendationsTuple[1]

		usedItems = self._getUsedItems(userId)


		self.log.info("Recommendations received in Ranker: %s" %recommendations)
		self.log.info("Recommendation types received in Ranker: %s" %recommendations.keys())
		results = []

		if "online" in recommendations: # online exists as long as user has been active
			results.extend(recommendations["online"][:self.numberToServe]) # should only has one

		if "offline" in recommendations: # offline exist only if user are registered, the recs could be from CF or LR
			results.extend(recommendations["offline"][:self.numberToServe])

		if "popular" in recommendations: # most popular should always exist
			# if there is no personalized recs, the remaining should be filled by most popular
			results.extend(recommendations["popular"][:self.numberToServe])
		else:
			self.log.error("recommendations do not contain popular items")

		try:
			# remove the already visited items
			results = np.random.choice(list(set(results)-usedItems), self.numberToServe, replace=False)
		except ValueError:
			# sometimes the user may watched a lot
			# this is apparently not a good strategy, why?
			results = np.random.choice(results, self.numberToServe, replace=False)


		return results

if __name__=="__main__":
	from DatabaseInterface import DatabaseInterface
	db = DatabaseInterface("DATA")
	db.startEngine()
	ranker = Ranker(numberToServe=10, database=db)
	print sorted(ranker._getUsedItems(1))