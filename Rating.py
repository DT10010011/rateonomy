
def ROIRating(roi_app):
    payback = roi_app.payback_years
    if payback is None:
        return  0 
    elif payback <= .5:
        return 1
    elif payback < 2:
        return 2
    elif payback < 3:
        return 3
    elif payback < 4:
        return 4
    elif payback < 5:
        return 5
    elif payback < 6:
        return 6
    elif payback < 7:
        return 7
    elif payback < 8:
        return 8
    elif payback < 9:
        return 9
    else:
        return 10              
        

def fieldSize(fieldsize):
    size = drivability_app.size
    ratedsize = (size/1000)*10
    return ratedsize

def fieldScore(fieldscore):
    ratings = [(98, 10), (96, 9), (94, 8), (92,7), (90, 6), (88, 5), (86, 4), (84, 3), (82, 2),(0, 1)]
    score = drivability_app.score
    for limit, rating in ratings[1:]:
        if score > limit:
            return rating

def drivabilityscore(fieldsize, fieldscore):
    size = fieldsize
    score = fieldscore
    return ((size * .1) + (score * .9)) / 2

