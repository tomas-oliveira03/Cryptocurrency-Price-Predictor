import datetime
from enum import Enum

class CronExpression(Enum):
    EVERY_SECOND = 1
    EVERY_2_SECONDS = 2
    EVERY_5_SECONDS = 5
    EVERY_10_SECONDS = 10
    EVERY_30_SECONDS = 30
    
    EVERY_MINUTE = 1 * 60
    EVERY_2_MINUTES = 2 * 60
    EVERY_5_MINUTES = 5 * 60
    EVERY_10_MINUTES = 10 * 60
    EVERY_30_MINUTES = 30 * 60
    
    EVERY_HOUR = 1 * 60 * 60
    EVERY_2_HOURS = 2 * 60 * 60
    EVERY_5_HOURS = 5 * 60 * 60
    EVERY_6_HOURS = 6 * 60 * 60
    EVERY_12_HOURS = 12 * 60 * 60
    
    EVERY_DAY = 1 * 24 * 60 * 60


# Calculates seconds until X time interval (aligned with PC clock)
# Example:
# If current time is 12:07:32 and cronInterval is EVERY_10_MINUTES (600 seconds),
# the function returns the number of seconds until 12:10:00 → 148 seconds.
def getSecondsUntilNextAlignedMark(cronInterval: CronExpression) -> int:
    now = datetime.datetime.now()
    totalSeconds = int(now.timestamp())
    intervalSeconds = cronInterval.value
    remainder = totalSeconds % intervalSeconds
    waitSeconds = (intervalSeconds - remainder) if remainder != 0 else 0
    return waitSeconds