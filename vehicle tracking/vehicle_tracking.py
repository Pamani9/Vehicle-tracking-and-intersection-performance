import cv2
from pathlib import Path
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from time import sleep
from tqdm import tqdm
import configparser
from scipy.spatial.distance import cdist
import pandas as pd




# background_mode = "realtime_MOG2" # prerun_MOG2, realtime_MOG2, median, 
cars_to_track = []
with open('cars_to_track.csv', 'r') as f:
    cars_to_track = [int(line[:-1]) for line in f]


def getBackground(background_mode, video_file, history=500, background_image='background.png', save_background=False):
    print(f'Defining the background using the {background_mode} method...')
    video_file = Path(video_file)
    background_image = Path(background_image)
            
    kernel3 = np.ones((3, 3), 'uint8')
    kernel5 = np.ones((5, 5), 'uint8')
     
    def fixedForeground(frame):
        fgd = cv2.absdiff(frame, background)
        fgd = cv2.cvtColor(fgd, cv2.COLOR_BGR2GRAY)
        _, fgd = cv2.threshold(fgd, 18, 255, cv2.THRESH_BINARY)
        cv2.morphologyEx(src=fgd, dst=fgd, op=cv2.MORPH_DILATE, kernel=kernel3)

        return fgd
    
    def MOG2Foreground(frame):
        fgd = fgbg.apply(frame)
        cv2.morphologyEx(src=fgd, dst=fgd, op=cv2.MORPH_OPEN, kernel=kernel3)
        cv2.morphologyEx(src=fgd, dst=fgd, op=cv2.MORPH_CLOSE, kernel=kernel3)
        return fgd
    
    if background_mode == 'median':
        cap = cv2.VideoCapture(str(video_file))
        FOI = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=history)
    
        frames = []
        for frameOI in FOI:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
            ret, frame = cap.read()
            frames.append(frame)
        
        background = np.median(frames, axis=0).astype(dtype=np.uint8)
        if save_background: cv2.imwrite(str(background_image), background)
        
        getForeground = fixedForeground
        
    if background_mode == 'prerun_MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2(history = history, detectShadows=True)
        fgbg.setShadowValue(0)

        cap = cv2.VideoCapture(str(video_file))
        framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ret, frame = cap.read()
        
        bar = tqdm(total=framecount, position=0, leave=True, desc="MOG2")
        while ret:
            bar.update(1)
            fgbg.apply(frame)
            ret, frame = cap.read()
        cap.release()
        
        background = fgbg.getBackgroundImage()
        if save_background: cv2.imwrite(str(background_image), background)
        
        getForeground = MOG2Foreground

    if background_mode == 'realtime_MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2(history = history, detectShadows=True)
        fgbg.setShadowValue(0)
        
        cap = cv2.VideoCapture(str(video_file))
        ret, frame = cap.read()
        background = frame # first frame as temporary background
        cap.release()
        
        getForeground = MOG2Foreground

    if background_mode == 'load_image':
        print(f'Loading {background_image.resolve()}')
        background = cv2.imread(str(background_image))
        getForeground = fixedForeground
        
    return background, getForeground
        

def frametime(video_file, timebox, display=True, expected_fps=30):
    video_file = Path(video_file)
    sleep(0.5)
    plt.close('all')
    cap = cv2.VideoCapture(str(video_file))
    
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), position=0, leave=True, desc="Getting frametime")
    
    ret, old_frame = cap.read()
    old_crop = old_frame[timebox[1]:timebox[3], timebox[0]:timebox[2],:]
    ret, frame = cap.read()
    
    values = []
    
    i=0
    while ret:
        i+=1
        if i%10 == 0: pbar.update(10)
        
        crop = frame[timebox[1]:timebox[3], timebox[0]:timebox[2],:]
        
        diff = cv2.absdiff(crop, old_crop)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        values.append(np.count_nonzero(thresh))
             
        if display:
            cv2.imshow("crop", thresh)
            key = cv2.waitKey(2)
            if key == 27 : break

        old_crop = crop
        ret, frame = cap.read()

    pbar.close()
    print(i)
    cv2.destroyAllWindows()
    
    nb_frames = len(values)    
    print(f"nb_frames: {nb_frames}")

    values = 10*(np.array(values)>=5)
    peaks, properties = signal.find_peaks(values, height=5, distance = 0.4*expected_fps)
    
    true_peaks = [peaks[0]]

    for i in range(len(peaks)-1):
        if peaks[i+1] - true_peaks[-1] > 20: # > 5
            true_peaks.append(peaks[i+1])
    peaks = true_peaks

    #print(peaks)

    fig, axes = plt.subplots(2,2, figsize=(15,9))
    gs = axes[0, 0].get_gridspec()
    for ax in axes[0,:]: ax.remove()
    axbig = fig.add_subplot(gs[0,:])
    
    axbig.plot(np.arange(len(values)), values)
    axbig.plot(peaks, np.array(values)[peaks], 'ro')

    axbig.set_xlabel("frame")
    axbig.set_ylabel("frame to frame change")

    frame2time = np.zeros(nb_frames, dtype=float)
    for i in range(len(peaks)-1):
        p1 = peaks[i]
        p2 = peaks[i+1]
        frame2time[p1:p2+1] = np.linspace(i, i+1, p2-p1+1)
        pass

    axes[1,0].plot(np.arange(len(frame2time)), frame2time)
    axes[1,0].set_xlabel("frame")
    axes[1,0].set_ylabel("time")
    
    delta = [frame2time[x+1]-frame2time[x] for x in range(peaks[0],peaks[-1])]
    axes[1,1].plot(np.arange(len(delta)), delta)
    axes[1,1].set_xlabel("frame")
    axes[1,1].set_ylabel("frametime")
    
    plt.show()
       
    
    fps = (peaks[-1]-peaks[0])/(frame2time[peaks[-1]]-frame2time[peaks[0]])
    
    print(f'Video FPS: {fps}')
    
    return frame2time, [peaks[0], peaks[-1]], fps

def box(event, x, y, flags, param):
    [frame, message] = param
    global x_start, x_end, y_start, y_end, img

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x,y
        img = frame.copy()   #pour réinitialiser l'image lorsqu'on trace une autre fenetre
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x,y
        cv2.rectangle(img,(x_start,y_start),(x_end,y_end),(0,255,0),5)
        cv2.imshow(message, img)

def get_box(frame, message):
    #[x_start, y_start, x_end, y_end] = [0,0,0,0]
    print(frame.shape)
    param = [frame, message]
    img = frame.copy()
    cv2.namedWindow(message)
    
    cv2.setMouseCallback(message, box, param)
    cv2.imshow(message, img)

    while(1):
        #cv2.rectangle(img,(x_start,y_start),(x_end,y_end),(0,255,0),5)
        #cv2.imshow('image', img)
        key = cv2.waitKey(10)

        if key == 13:
            break
        elif key == 27:
            return False
        
    cv2.destroyAllWindows()
    x1 = min([x_start,x_end])
    x2 = max([x_start,x_end])
    y1 = min([y_start,y_end])
    y2 = max([y_start,y_end])
    return [x1, y1, x2, y2]
    
def isInBBox(point, bbox):
    [x, y] = point
    [x1, y1, x2, y2] = bbox
    
    if x1<x and x<x2 and y1<y and y<y2:
        return True
    else:
        return False
    
def getTime(video_file, background, expected_fps=30, useClock=False):
    video_file = Path(video_file)
    if useClock:
        print('Please select clock window')
        timeBox = get_box(background, 'time box selection')
        frame2time, frameBounds, fps = frametime(video_file, timeBox, display=False, expected_fps=expected_fps)
        return frame2time, frameBounds, fps
    else:
        cap = cv2.VideoCapture(str(video_file))
        framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame2time = np.arange(framecount)/expected_fps
        frameBounds = [100, framecount-10]
        return frame2time, frameBounds, fps

class EMA:
    def __init__(self, window, fps, refreshPeriod, doubleEMA=False):
        self.fps = fps
        self.window = window
        self.tau = self.window*self.fps
        self.alpha = 2/(self.tau+1)
        self.internalEMA = 0
        self.displayedEMA = self.internalEMA
        self.fullAverage = self.internalEMA
        self.max = self.internalEMA
        self.timer = 0
        self.refreshPeriod = refreshPeriod
        self.count = 0
        self.total_time = 0
        self.doubleEMA = doubleEMA
        
        if doubleEMA:
            self.tau2 = self.refreshPeriod*self.fps*4
            self.beta = 2/(self.tau2+1)
            self.internalEMA2 = 0
    
    def update(self, newCars): # To be called on every frame
        self.count += newCars
        if self.total_time == 0 and newCars==0:
            self.total_time = 0
        else: self.total_time += 1/self.fps
        self.timer += 1/self.fps
        
        self.internalEMA = newCars*self.alpha*self.fps + self.internalEMA*(1-self.alpha)
        
        if self.doubleEMA:
            self.internalEMA2 = self.internalEMA*self.beta + self.internalEMA2*(1-self.beta)
            if self.internalEMA2 > self.max: self.max = self.internalEMA2
            
            
        else:
            if self.internalEMA > self.max: self.max = self.internalEMA
        
        
        
        if self.timer >= self.refreshPeriod:
            if self.doubleEMA: self.displayedEMA = self.internalEMA2
            else: self.displayedEMA = self.internalEMA
            self.timer = 0
            if self.total_time > self.window*1.5:
                self.fullAverage = self.count / self.total_time
        
    def get_EMA(self):
        return self.displayedEMA
    def get_fullAverage(self):
        return self.fullAverage
    def get_max(self):
        return self.max
        
def contourCenter(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return [cx, cy]
    else:
        return [0,0]
    
def displayContourShape(l_contours, image, BBox=None, color=(0,0,255), thickness=1):
    if BBox == None:
        for contour in l_contours: cv2.drawContours(image, contour, -1, color, thickness)
    else:
        for contour in l_contours:
            centroid = contourCenter(contour)
            if isInBBox(centroid, BBox):
                cv2.drawContours(image, contour, -1, color, thickness)

    
def displayContourRect(l_contours, image, BBox=None, color=(0,255,0), thickness=2):
    if BBox == None:
        for contour in l_contours:   
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, color, thickness)
            
            #cv2.rectangle(image, (x1, y1), (x2, y2),color, thickness)
    else:
        for contour in l_contours:
            centroid = contourCenter(contour)
            if isInBBox(centroid, BBox):
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, color, thickness)

def matching2(centroids, car_list):
    carIds = np.array([car.id for car in car_list if car.visible])
    carCoords = np.array([car.expected_position() for car in car_list if car.visible])
    
    nb_contours = len(centroids)
    matched_id = np.array([-1]*nb_contours)
    
    if carCoords.size !=0: # Skip this if we don't already have cars
        #carIsMatched = np.zeros(carCoords.shape[0], dtype=bool) # True if a contour has been matched with this car
        dmatrix = cdist(carCoords, centroids)
        closest2car = np.argmin(dmatrix, axis=1)
        #sec_closest2car = np.squeeze(np.argpartition(dmatrix, 1, axis=1)[:,1])
        
        for contour_id in range(nb_contours):
            cars_as_closest = np.where(closest2car == contour_id)[0] # Cars that have this centroid as closest
            
            if len(cars_as_closest) == 1: # Only one existing car has this centroid as closest
                visi_car_id = cars_as_closest[0]
                matched_id[contour_id] = carIds[visi_car_id]
                
            elif len(cars_as_closest) > 1: # More than one car has this centroid as closest
                matched_id[contour_id] = -9999
            
            else: # No car is within distance of this centroid
                matched_id[contour_id] = -1
    
    return matched_id


class Car:
    def __init__(self, id_, coordinates, entryCard, entryTime):
        self.id = id_
        self.coordinates = coordinates
        self.visible = True
        self.speed = [0,0]
        self.EMAspeed = [0,0]
        self.EMAalpha = 2/(5+1)
        self.idle = 0
        self.entryCard = entryCard
        self.entryTime = entryTime
        self.exitCard = "none"
        self.exitTime = 0
        
    
    def distance(self, location):
        return ((location[0] - self.coordinates[0])**2 + (location[1] - self.coordinates[1])**2)**0.5

    def delta(self, location):
        [x, y] = self.expected_position()

        return ((location[0] - x)**2 + (location[1] - y)**2)**0.5
    
    def calculate_speed(self, location):
        self.speed[0] = (location[0] - self.coordinates[0])/self.idle
        self.speed[1] = (location[1] - self.coordinates[1])/self.idle
        
        if self.EMAspeed == [0,0]:
            self.EMAspeed[0] = self.speed[0]
            self.EMAspeed[1] = self.speed[1]
        else:    
            self.EMAspeed[0] = self.speed[0]*self.EMAalpha + self.EMAspeed[0]*( 1-(self.EMAalpha)**self.idle )
            self.EMAspeed[1] = self.speed[1]*self.EMAalpha + self.EMAspeed[1]*( 1-(self.EMAalpha)**self.idle )

    def scalar_speed(self):
        return (self.speed[0]**2 + self.speed[1]**2)**0.5
    
    def expected_position(self):
        x = self.coordinates[0] + self.speed[0]*(self.idle+1)
        y = self.coordinates[1] + self.speed[1]*(self.idle+1)
        return [x, y]
    
    def info(self):
        print(f'Car n° {self.id}')
        print(f'Entry point: {self.entry}')
        print(f'Exit point: {self.exit}')
        print(f'x : {self.coordinates[0]}')
        print(f'y : {self.coordinates[1]}')
        if self.visible: print('Car is visible')
        else: print('Car is not visible')
        print(f'Speed: {self.speed} pixel/frame')
        print(f'Idle for {self.idle} frames')
    

def update_cars_positions(car_list, centroids, matched_id, detection_bbox, currentTime):
    [WB, NB, EB, SB] = detection_bbox
    quadrant = {0:"South",
                1:"East",
                2:"West",
                3:"North"}
    
    for i, (centroid, car_id) in enumerate(zip(centroids, matched_id)):
        
        if car_id >= 0: # If centroid matches an existing car, update it
            car_list[car_id].calculate_speed(centroid)
            car_list[car_id].coordinates = centroid
            car_list[car_id].idle = 0
        
        elif car_id == -1: # If centroid doesnt't match an existing car, create new car
            if centroid[0]<EB and centroid[0]>WB and centroid[1]>NB and centroid[1]<SB:
                id_ = len(car_list)
                coordinates = centroid
                x, y = coordinates[0], coordinates[1]
                switch=0

                if (x-WB)/(EB-WB) > (y-NB)/(SB-NB) : switch+=1
                if (x-WB)/(EB-WB) < 1 - (y-NB)/(SB-NB) : switch+=2
                
                entryCard = quadrant[switch]
                entryTime = currentTime
                car_list.append(Car(id_, coordinates, entryCard, entryTime))


def update_cars_status(car_list, detection_bbox, currentTime, myEMA, max_idle=50):
    [WB, NB, EB, SB] = detection_bbox
    newly_exited=0
    for car in car_list:
        if car.visible:
            [x,y] = car.coordinates
            [vx, vy] = car.EMAspeed
            
            car.idle += 1

            # Remove lost cars (idle for too long)
            if car.idle >= max_idle : 
                car.visible = False
                car.exitCard = 'Lost'
                #del visibleCars_id2coord[car.id]

            # Remove cars that exited the detection box
            if x<WB and car.speed[0]<0:
                car.visible = False
                car.exitCard = 'West'
                car.exitTime = currentTime
                newly_exited+=1
                #del visibleCars_id2coord[car.id]
            elif x>EB and car.speed[0]>0:
                car.visible = False
                car.exitCard = 'East'
                car.exitTime = currentTime
                newly_exited+=1
                #del visibleCars_id2coord[car.id]
            elif y>SB and car.speed[1]>0:
                car.visible = False
                car.exitCard = 'South'
                car.exitTime = currentTime
                newly_exited+=1
                #del visibleCars_id2coord[car.id]
            elif y<NB and car.speed[1]<0:
                car.visible = False
                car.exitCard = 'North'
                car.exitTime = currentTime
                newly_exited+=1
                #del visibleCars_id2coord[car.id]
    myEMA.update(newly_exited)



def displayVisibleCars(image, car_list, color=(0,255,255)):
    y_max = image.shape[0]
    for car in car_list:
        if car.visible:
            [x,y] = car.coordinates
            
            if car.id in cars_to_track:
                cv2.putText(image, f"{car.id}", (max(int(x)-10,0), min(int(y)+10, y_max)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,0), 2)
                continue
            cv2.putText(image, f"{car.id}", (max(int(x)-10,0), min(int(y)+10, y_max)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

def displayCarVectors(image, car_list, color=[255,128,128], speed_scale_factor=3):
    f = speed_scale_factor
    for car in car_list:
        if car.visible:
            if car.idle <= 1:
                [x,y] = car.coordinates
            else:
                [x,y] = car.expected_position()
            [vx, vy] = car.EMAspeed
            cv2.circle(image, (int(x),int(y)), 5, color, 3)
            cv2.line(image, (int(x),int(y)), (int(x+f*vx),int(y+f*vy)), color, 3)

def displayLostCars(image, car_list, color=(0,0,255)):
    for car in car_list:
        if car.exitCard == 'Lost':
            [x,y] = car.coordinates
            cv2.putText(image, f"car {car.id}", (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)  


def logInLoop(car_list, currentTime, logFile):
    pass
def logCarsInfo(enableLogCarsInfo = False, logFile = 'car_log.txt', parameters = {}):
    logFile = Path(logFile)  
    
    if enableLogCarsInfo:
        print(f'Logging car infos to {logFile.resolve()}')
        with logFile.open('w') as f:
            f.write('[Parameters]\n')
            for k, v in parameters.items():
                f.write(f'{k}: {v}\n')
            f.write('\n')
            f.write('[FORMAT]\n')
            f.write('id;coordinates;speed;EMAspeed')
            f.write('\n')
            f.write('[DATA]\n')
        def logInLoop(car_list, currentTime, logFile):
            with logFile.open('a') as f:
                f.write(f'{currentTime}\t')
                f.write('\t'.join([f'{car.id};{car.coordinates};{car.speed};{car.EMAspeed}' for car in car_list if car.visible]))
                f.write('\n')
        return logFile, logInLoop
    
    else:
        return None, logInLoop
        
        
def displayBBox(image, detection_bbox, color=(255, 0, 0)):
    [WB, NB, EB, SB] = detection_bbox
    cv2.rectangle(image, (WB, NB), (EB, SB), color, 2)

def Wait(delay=5):
    global isPaused
    key = cv2.waitKey(delay)
    quit_p = 0
    if key == ord('p') or isPaused==1:
        key2 = cv2.waitKey(-1)
        if key2 == ord('p'): isPaused=1
        elif key2 == (ord('q')): quit_p=1
        else:
            print("a")
            isPaused=0
    if key == 27 or quit_p==1:
        return True
    return False

def Display(image, display_size, outVideo=None):
    cv2.imshow("single", cv2.resize(image, display_size))
    if outVideo is not None: outVideo.write(cv2.resize(image, display_size))
    return Wait()

def main_loop(video_file, frame2time, frameBounds, getForeground, detection_bbox, myEMA,
              detection_size=100, display_size = (1280,720), drawContourShape=True, drawContourRect=True,
              showVisibleCars=True, showCarVectors=True, showLostCars=False, logFile='car_log.txt', logInLoop=logInLoop):
    
    video_file = Path(video_file)
    cap = cv2.VideoCapture(str(video_file))
    global isPaused
    isPaused=0
    
    ret, frame = cap.read()
    original = frame.copy()
    
    car_list = []
    #visibleCars_id2coord = {}
    #depth = []
    
    bar = tqdm(total=frameBounds[1], position=0, leave=True, desc="Vehicle detection")

    for currentFrame in range(frameBounds[1]):
        ret, frame = cap.read()
        if not ret:
            break
        
        bar.update(1)
        currentTime = frame2time[currentFrame]
        foreground = getForeground(frame)
        
        if currentFrame < frameBounds[0]:
            continue
        if currentFrame >= frameBounds[1]:
            break
        
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2RGB)
        
        contours = [contour for contour in contours if cv2.contourArea(contour) > detection_size]
        
        centroids = [contourCenter(contour) for contour in contours]
        
        if drawContourShape: displayContourShape(contours, foreground, BBox=None)
        if drawContourRect: displayContourRect(contours, frame, BBox=None)
        
        [WB, NB, EB, SB] = detection_bbox
        
        matched_id = matching2(centroids, car_list)
        update_cars_positions(car_list, centroids, matched_id, detection_bbox, currentTime)
        update_cars_status(car_list, detection_bbox, currentTime, myEMA, max_idle=50)
        logInLoop(car_list, currentTime, logFile)
        
        if showVisibleCars: displayVisibleCars(frame, car_list, color=(0,255,255))
        if showCarVectors: displayCarVectors(foreground, car_list, color=[255,128,128], speed_scale_factor=3)
        if showLostCars: displayLostCars(foreground, car_list, color=(0,0,255))
        
        displayBBox(frame, detection_bbox, color=(255, 0, 0))
        
        if Display(foreground, display_size): break
    cap.release()
    bar.close()
    return car_list, currentTime

def printStats(car_list, frame2time, currentTime, min_validDelay=0.1, result_file=None):
    
    dataframe = [ [car.id, car.entryCard, car.exitCard, car.entryTime, car.exitTime] for car in car_list]
    car_df = pd.DataFrame(dataframe, columns=["Car",
                                              "Cardinal of entry",
                                              "Cardinal of exit",
                                              "Time of entry",
                                              "Time of exit"])
    car_df["Delay"] = car_df["Time of exit"] - car_df["Time of entry"]
    
    entered_condition = car_df["Time of entry"]>0.01
    exited_condition = car_df["Cardinal of exit"].isin(["North", "South", "East", "West"])
    delay_condition = car_df["Delay"] > min_validDelay
    
    total_time = currentTime-(car_df[entered_condition]["Time of entry"]).min()
    throughput = exited_condition.sum(axis=0)/total_time
    print('\n')
    print(f"Number cars that went through: {np.count_nonzero(entered_condition & exited_condition & delay_condition)}")
    print(f"Number of lost cars: {np.count_nonzero(car_df['Cardinal of exit']=='Lost')}")
    print(f"Number of cars who disappeared within {round(min_validDelay, 3)}s: {np.count_nonzero(entered_condition & exited_condition & (1-delay_condition))}")
    print('\n')
    print(f"\nIntersection throughput : {round(throughput,2)} veh/s (or {round(60*throughput)} veh/min) for {total_time} seconds")
    
    df_OD = car_df[entered_condition & exited_condition & delay_condition]
    
    if result_file is not None:
        result_file = Path(result_file)
        df_OD.to_csv(result_file.parent/(result_file.stem+'_valid'+result_file.suffix), index=False)
        car_df.to_csv(result_file.parent/(result_file.stem+'_all'+result_file.suffix), index=False)

if __name__ == '__main__':
    cv2.destroyAllWindows()
    plt.close('all')
    
    config = configparser.ConfigParser()
    config.read('configFile.ini')
    
    folder = Path(config.get('main', 'folder'))
    video_file = folder / config.get('main', 'video_file')
    
    
    background_mode = config.get('main', 'background_mode')
    
    background_image='background.png'
    if background_mode == 'load_image':
        background_image = config.get('main', 'background_image')
        
    background_history = int(config.get('main', 'background_history'))
    background, getForeground = getBackground(background_mode, video_file, background_image=background_image, history=background_history)
    
    expected_fps = float(config.get('main', 'expected_fps'))
    useClock = (config.get('main', 'useClock')).lower() in ('true', '1', 'y', 'yes')
    frame2time, frameBounds, fps = getTime(video_file, background, expected_fps, useClock)
    
    # print('Please select detection window')
    # [WB, NB, EB, SB] = get_box(background, 'detection box')
    [WB, NB, EB, SB] = [796, 301, 1292, 798]
    center = [int((NB+SB)/2), int((WB+EB)/2)]
    detection_bbox = [WB, NB, EB, SB]
    
    MA_window = 10
    refreshPeriod = 0.25
    myEMA = EMA(MA_window, fps, refreshPeriod, doubleEMA=True)
    
    parameters = {'video_file':video_file.name,
                  'background_mode':background_mode,
                  'background_history':background_history,
                  'useClock':useClock,
                  'expected_fps':expected_fps}
    logFile, logInLoop = logCarsInfo(enableLogCarsInfo = True, logFile = 'car_log.txt', parameters = parameters)
    
    car_list, currentTime = main_loop(video_file, frame2time, frameBounds, getForeground, detection_bbox, myEMA, logFile=logFile, logInLoop=logInLoop)
    printStats(car_list, frame2time, currentTime, result_file='car_OD.csv')
    
    