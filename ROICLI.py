#ROI calculator using the command line interface.

#Calculate cost for conventional operations
#ConEquipment Cost
conEquipCost = int(input("Enter purchase price for a new machine or amount left on equipment loan"))
conSubCost = int(input("Enter price for subcriptions used on your conventional fleet. This includes products such as autosteer, RTK, and anyother added subscriptions."))
conTotalCost = conEquipCost+conSubCost 
print("contotalcost", conTotalCost)
#ConLabor Cost
conNumMachines = int(input("Enter the number of machines in your operation that could be autonomy enabled"))
conLaborRate = float(input("Enter labor rate ($/Hr)"))
conLaborCost = conNumMachines*conLaborRate
print("conlaborcost", conLaborCost)
#Calculate cost for autonomus operations

#AutEqiupment Cost
autEquipCost = int(input("Enter purchase price for a new machine or amount left on equipment loan"))
autRetroCost = int(input("Enter price to install autnomy kit if required. If none click enter to skip.") or 0)
autSubCost = int(input("Enter price for required subscriptions. This includes products such as autosteer, RTK, and any other added subscriptions."))
autTotalCost = autEquipCost+autRetroCost+conSubCost
print("auttotalcost", autTotalCost) 
#AutLabor Cost
autNumMachines = int(input("Enter the number of machines in your operation that could be autonomy enabled"))
autMachineperOperator = int(input("Enter number of machines per operator"))
dedicatedTimeRaw = int(input("Enter the percentage of time operator speds supervising the equipment compared to other task(1-100)"))
dedicatedTimeProcessed = dedicatedTimeRaw/100
autLaborRate = int(input("Enter labor rate ($/Hr)"))
autLaborCost = (autLaborRate*dedicatedTimeProcessed)/autNumMachines
print("autLaborCost", autLaborCost)
#Savings From Autonomy
savings = conLaborCost-autLaborCost
print("Savings",savings)
#Payback period
costdif = conTotalCost-autTotalCost
payback = costdif/savings

print("Payback",payback)