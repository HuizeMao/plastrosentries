#import necessary libraries
import picamera
from logzero import logger
from math import sin,cos
from time import sleep
from datetime import datetime, timedelta
import logging
import logzero
import random
from ephem import readtle, degree
import math
import csv
import os 

###initializations

#define directory path of this file
dir_path = os.path.dirname(os.path.realpath(__file__))
#function that creates a data file
def create_csv_file(data_file):
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Latitude ", "Longitude")
        writer.writerow(header)

#function that add data to the data file
def add_csv_data(data_file, data):
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)



###capture images with position labels
##init
# Set up camera
cam = picamera.PiCamera()
cam.resolution = (1296, 972)

# Latest TLE data for ISS location
name="ISS (ZARYA)"             
l1="1 25544U 98067A   20033.25120128  .00002314  00000-0  49941-4 0  9990"
l2="2 25544  51.6428 301.0629 0005407 220.6572 248.5738 15.49135306210949"
iss = readtle(name, l1, l2)
iss.compute()

##find position of ISS
def get_latlon():
    """
    A function that writes ISS postion to EXIF data for satellite images
    also returns the lat and long it calculates
    """
    iss.compute() # Get the lat/long values from ephem

    ##get the longitutude in the format of degree:min:sec
    long_value = [float(i) for i in str(iss.sublong).split(":")] 
    
    if long_value[0] < 0: #if the degree of longitude is below 0, the ISS is at the West of the prime meridian
        long_value[0] = abs(long_value[0]) 
        cam.exif_tags['GPS.GPSLongitudeRef'] = "W"
    else: #if it is greater than or equal to zero,the ISS is at the East of the prime meridian
        cam.exif_tags['GPS.GPSLongitudeRef'] = "E"
    #label the saellie image with ISS's longitude position
    cam.exif_tags['GPS.GPSLongitude'] = '%d/1,%d/1,%d/10' % (long_value[0], long_value[1], long_value[2]*10) 


    ##get the lattitude in the format of degree:min:sec 
    lat_value = [float(i) for i in str(iss.sublat).split(":")]

    if lat_value[0] < 0: #if the degree of lattitudes is less than 0, we are at the South of Equator
        lat_value[0] = abs(lat_value[0])
        cam.exif_tags['GPS.GPSLatitudeRef'] = "S"
    else:#if the degree of lattitudes is greater than or equal to 0, we are at the North of Equator
        cam.exif_tags['GPS.GPSLatitudeRef'] = "N"
    #label satellite image with ISS's lattitude position
    cam.exif_tags['GPS.GPSLatitude'] = '%d/1,%d/1,%d/10' % (lat_value[0], lat_value[1], lat_value[2]*10)
    #for debug: print(str(lat_value), str(long_value))
    return (iss.sublat / degree, iss.sublong / degree)




###run for three hours
## initialise the CSV file
data_file = dir_path + "/PSdata.csv"
create_csv_file(data_file)
# store the start time
start_time = datetime.now()
# store the current time
now_time = datetime.now()
photo_counter=1

##collect data for roughly three hours
while (now_time < start_time + timedelta(minutes=178)):
    try:
        logger.info("{} iteration {}".format(datetime.now(), photo_counter))
        # get latitude and longitude
        lat, lon = get_latlon()
        lat = round(lat,3)
        lon = round(lon,3)
        # Save the data to the file
        data = (lat, lon)
        add_csv_data(data_file, data)
        # use zfill to pad the integer value used in filename to 4 digits (e.g. 0001, 0002...)
        cam.capture(dir_path + "/photo_" + str(photo_counter).zfill(4) + ".jpg")
        photo_counter += 1
        #sleep
        sleep(8)
        # update the current time
        now_time = datetime.now()
    except Exception as e:
        logger.error('{}: {})'.format(e.__class__.__name__, e))
