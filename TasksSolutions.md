# Tasks Expected Solutions

StairClimber:

Maze:

FourCorners:
DEF run m( WHILE c( noMarkersPresent c) w( WHILE c( frontIsClear c) w( move w) putMarker turnLeft move w) m)

TopOff: 
DEF run m( WHILE c( frontIsClear c) w( IF c( markersPresent c) i( putMarker i) move w) m)

Harvester:
Leaps Behaviour: DEF run m( turnLeft turnLeft WHILE c( frontIsClear c) w( move w) turnLeft turnLeft WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)
Crashable: DEF run m( turnLeft turnLeft WHILE c( frontIsClear c) w( move w) turnLeft turnLeft WHILE c( markersPresent c) w( WHILE c( frontIsClear c) w( pickMarker move w) pickMarker turnLeft move turnLeft WHILE c( frontIsClear c) w( pickMarker move w) pickMarker turnRight move turnRight w) m)

CleanHouse:
Leaps Behaviour: DEF run m( WHILE c( noMarkersPresent c) w( IF c( leftIsClear c) i( turnLeft i) move IF c( markersPresent c) i( pickMarker i) w) m)
Crashable: DEF run m( WHILE c( noMarkersPresent c) w( IF c( leftIsClear c) i( turnLeft i) IFELSE c( frontIsClear c) i( move i) ELSE e( turnRight turnRight e) IF c( markersPresent c) i( pickMarker i) w) m)
