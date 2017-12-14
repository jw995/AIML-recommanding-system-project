# Database Interface
# to simulate some database operations

import os
import pandas as pd
import logging

class DatabaseInterface(object):
	logging.basicConfig(level=logging.INFO)

	# in reality, it should be a configuration file
	HISTORY = "ratings.csv"
	USER_FEATURE = "userFeature.csv"
	ITEM_FEATURE = "itemFeature.csv"
	INVENTORY = "inventory.csv" #in reality, inventory store all the representations, such as video link

	HISTORY_KEY = "history"
	USER_FEATURE_KEY = "user_feature"
	ITEM_FEATURE_KEY = "item_feature"
	INVENTORY_KEY = "inventory"
	USER_ACTIVITY_KEY = "user_activity"

	# register the static database first
	dbTable = {HISTORY_KEY: HISTORY,
			   USER_FEATURE_KEY: USER_FEATURE,
			   ITEM_FEATURE_KEY: ITEM_FEATURE,
			   INVENTORY_KEY: INVENTORY}

	def __init__(self, path):
		self.log = logging.getLogger(__name__)
		self.path = path
		self.started = False
		self.connTable = {}

	def startEngine(self):
		if self.started:
			self.log.warning("the data base has already started")
			# start a running engine is not permitted here since it will remove all unsaved data
		else:
			self.log.info("start the database engine...")
			for tableName, tablePath in self.dbTable.iteritems():
				self.log.info("loading table %s..." % tableName)
				self.connTable[tableName] = pd.read_csv(os.path.join(self.path, tablePath), index_col=0)

			self.log.info("creating table user_activity...")
			self.connTable[self.USER_ACTIVITY_KEY] = self.connTable["history"].groupby("user_id").size() # actually a series

			self.log.info("database successfully started")
			self.started = True

	# ideally a sql should be used to query a database, in this case, pandas operation will used instead in client
	# https://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html
	def extract(self, tableName):
		return self.connTable[tableName]

	def putAction(self, action):
		insertRow(self.connTable[self.HISTORY_KEY], [action.userId, action.itemId, action.rating])

def insertRow(df,row):
	# unsafe insertion into pandas dataframe
	df.loc[len(df)] = row


if __name__ == "__main__":
	connector = DatabaseInterface("DATA")
	connector.startEngine()
	df1 = connector.connTable["history"]
	print df1.head()
	df2 = connector.connTable["user_activity"]
	print df2[10]
	df3 = connector.connTable["item_feature"]
	print df3.loc[:,"unknown":]
	df4 = connector.connTable["user_feature"]
	print df4.loc[:,"age":]
	print set(df1[df1.loc[:,"user_id"]==2].loc[:,"item_id"])

