# main.py
# simulate different request coming into the system

from Webserver import WebServer, Request, Action

configMap = {"numberToServe": 10, "data_dir": "DATA"}
server = WebServer(configMap)
server.start() # load all the data in the database, start the first model training

# now experiment
reqX1 = Request(userId='X1') # anonymous user
req1 = Request(userId=1) # if it is a registered user, we use integer
print(reqX1)
print(req1)

recX1 = server.renderRecommendation(reqX1) # output recommendations
print recX1

rec1 = server.renderRecommendation(req1) # output recommendations
print(rec1)

# now we start an action
action1 = Action(1, 255, 5) # user 1 rated item 255 as score 5
print server.getFromInventory(255) # find out the name of item 255
server.getAction(action1) # feed the action to the server
rec1_afteraction = server.renderRecommendation(req1) # get recommendation after the system knows about the action
print(rec1_afteraction)

actionX1 = Action('X1', 123, 5) # anonymous user's action won't be saved in database
print server.getFromInventory(123)
server.getAction(actionX1)
recX1_afteraction = server.renderRecommendation(reqX1)
print(recX1_afteraction)

# update the system, e.g. one day has passed
server.increment()
# the system should forget about actionX1
recX1_aftercleaning = server.renderRecommendation(reqX1)
print(recX1_aftercleaning) # should be similar to recX1


req19 = Request(userId=19) # the one with very few history, so it is a new user
rec19 = server.renderRecommendation(req19)
print(rec19)