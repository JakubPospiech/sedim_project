from imanalyzer import Imanalyzer
import cv2
import cv2.ximgproc
import time
import numpy as np
class Histanalyzer(Imanalyzer):
    """Analysis of sedimentation process using histogram backprojection

    :attr type analyzer_type: Describes analyzer type for other objects (liveplot)
    :attr type lp: Method's iteration number
    :attr type roi_comparision: Coordinates of cropped image for model histogram
    :attr type roihist: Model histogram
    :attr type params: Parameters of blobs detector
    :attr type detector: Blobs detector
    :attr type set_up_blob_detection: Configuration of blob detector
    :attr type print_floc_surface: Flag, True when information should be printed, False otherwise

    """
    def __init__(self, roi_for_comparision):
        super(Histanalyzer, self).__init__()
        self.lp = 0
        self.analyzer_type = 3
        self.roi_comparision = roi_for_comparision
        self.roihist = 0
        self.params, self.detector = self.set_up_blob_detection()
        self.old_higher = 0
        self.new_higher = 0
        self.print_floc_surface = True

    def analyze(self, img, timestamp=0):
        """Histogram backprojection analyzer

        :param type img: Image of the settler after preprocessing
        :param type timestamp:
        :return: Height of sludge in percents of settler height
        0 - method doesn't evaluate these values
        """
        # Saving image after preprocessing
        cv2.imwrite("./step_out/" + str(self.lp) + "_step2.png", img)
        # RGB 2 HSV (calcBackProject uses HSV)
        hsvt = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # in first iteration model histogram is created
        if self.lp == 0:
            self.roihist = self.make_histogram(hsvt, img)
            print(self.roihist.size)
            print(self.roihist.shape)
            print(self.roihist[90])
            height, width, channels = img.shape
            if height > width:
                self.img_vertical = 1
        self.lp += 1
        # dst - mask of probability that certain pixel belongs to model area
        # "image in grayscale" - 0-255 (later threshold)
        dst = cv2.calcBackProject([hsvt],[0,1],self.roihist,[0,180,0,256],1)
        # Saving histogram backprojection results
        #dst_save = cv2.merge((dst, dst, dst))
        #cv2.imwrite("./step_out/step3_" + str(self.lp) + ".png", dst_save)
        # cropping to 99% of liquid height - easy way to exclude sludge floating on a surface
        # additionally cropping to level of bottom of the tank
        dst_height, dst_width = dst.shape
        index_99_per = self.top_of_the_tank + int((0.01 * (self.bottom_of_the_tank - self.top_of_the_tank)))
        dst[0:index_99_per, :] = np.zeros((index_99_per, dst_width), dtype=np.uint8)
        last_rows = self.bottom_of_the_tank - dst_height
        dst[last_rows:] = np.zeros((-last_rows, dst_width), dtype=np.uint8)
        dst_origin = dst
        dst_save = cv2.merge((dst, dst, dst))
        cv2.imwrite("./step_out/" + str(self.lp) + "_step3.png", dst_save)
        # edge detection with Canny algorithm
        # edges = cv2.Canny(dst, 0, 100)
        # Niblack's binarization
        edges = cv2.ximgproc.niBlackThreshold(dst, 255, cv2.THRESH_BINARY, 21, 0.25)

        # Saving Niblack's binarization results
        edges_img = cv2.merge((edges, edges, edges))
        cv2.imwrite("./step_out/" + str(self.lp) + "_step4.png", edges_img)

        # erosion with horizontal line to exclude any vertical-ish oriented edges
        if dst_width > 150:
            rectangle_short = cv2.getStructuringElement(cv2.MORPH_RECT, (dst_width // 150, 1))
            edges = cv2.erode(edges, rectangle_short)

        # Saving erosion results
        edges_img = cv2.merge((edges, edges, edges))
        cv2.imwrite("./step_out/" + str(self.lp) + "_step5.png", edges_img)

        # dilation with longer horizontal line to rebuild eroded horizontal edges
        # few times longer line is used to additionally highlight present horizontal lines
        rectangle_long = cv2.getStructuringElement(cv2.MORPH_RECT, (dst_width // 20, 1))
        edges = cv2.dilate(edges, rectangle_long)

        # Saving dilation results
        edges_img = cv2.merge((edges, edges, edges))
        cv2.imwrite("./step_out/" + str(self.lp) + "_step6.png", edges_img)

        # for presentation purpose only
        thresh = cv2.merge((edges, edges, edges))

        sediment_height, tank_width = self.pix_density(thresh)

        # detects large errors (gathers history of previous values)
        sediment_height = self.MedianFilter.correctError(sediment_height)

        # level of liquid in pixels
        liquid_level = self.bottom_of_the_tank - self.top_of_the_tank

        # result is converted to percents
        percent_out = sediment_height / liquid_level * 100

        # Determining if free floating flocs are present above the determined Sludge Blanket Height
        self.flocs_detection(dst_origin, sediment_height, tank_width, liquid_level)

        self._notify([percent_out, self.analyzer_type, timestamp])
        if self.write_debug_image:
            self.save_debug_image(thresh,liquid_level,percent_out)

        height, width, channels = dst_save.shape
        # dst_save = cv2.line(img,(0,self.top_of_the_tank),(width,self.top_of_the_tank),(255,0,0),3)
        # dst_save = cv2.line(img,(0,self.bottom_of_the_tank),(width,self.bottom_of_the_tank),(255,255,0),3)

        pixels_from_top = int((100-percent_out)*0.01*abs(self.bottom_of_the_tank-self.top_of_the_tank))
        dst_save = cv2.line(img,(0,self.top_of_the_tank+pixels_from_top),(width,self.top_of_the_tank+pixels_from_top),(0,255,0),3)

        # test_line_50per = int((50)*0.01*abs(self.bottom_of_the_tank-self.top_of_the_tank))
        # test_line_80per = int((100-80)*0.01*abs(self.bottom_of_the_tank-self.top_of_the_tank))
        # dst_save = cv2.line(img,(0,self.top_of_the_tank+test_line_50per),(width,self.top_of_the_tank+test_line_50per),(64,0,0),3)
        # dst_save = cv2.line(img,(0,self.top_of_the_tank+test_line_80per),(width,self.top_of_the_tank+test_line_80per),(128,0,0),3)

        cv2.imwrite("./hist_out/dst_" + str(self.lp) + "_" + str(percent_out) + ".png",dst_save)

        return 0, percent_out, 0

    def make_histogram(self,img,img_rgb):
        """
        Model histogram is created
        """
        img_tmp = img_rgb
        coord_x = (self.roi_comparision[0][0] , self.roi_comparision[1][0])
        coord_y = (self.roi_comparision[0][1] , self.roi_comparision[1][1])
        img = img[min(coord_y):max(coord_y), min(coord_x):max(coord_x)]

        cv2.rectangle(img_tmp, tuple(self.roi_comparision[0]), tuple(self.roi_comparision[1]), (0, 255, 0), 2)
        cv2.imwrite("./hist_out/ahist_roi_" + str(self.lp) + ".png",img_tmp)

        roihist = cv2.calcHist([img],[0, 1], None, [180, 256], [0, 180, 0, 256] )
        cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
        self.hist_comparision = roihist
        return self.hist_comparision

    def find_biggest_blob_diameter(self,thresh):
        keypoints = self.detector.detect(thresh)
        if self.write_debug_image:
            target_with_keypoints = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            timeString  = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            cv2.imwrite("./hist_out/hist_out_" + timeString + ".png",target_with_keypoints)
        diameter = 0
        for blob in keypoints:
            if diameter < blob.size:
                diameter = blob.size
        return diameter

    def pix_density(self, tresh):
        """
        Calculating height of sludge in pixels

        """
        # conversion to grayscale
        tresh = cv2.cvtColor(tresh, cv2.COLOR_BGR2GRAY)

        # sum of pixels in each column (because image is rotated -> settler "lies horizontally")
        # depends on self.vertical
        pix_dens = tresh.sum(axis=self.img_vertical, dtype=np.float32)

        # calculating width of tank on the image, doing that this way gives satisfactory accuracy
        tank_width = max(pix_dens) / 255

        # looking from top to bottom for top of the sludge
        pix_dens = (pix_dens / (max(pix_dens) + 1)) * 255
        height = np.nonzero(pix_dens > 75)
        correct_height = height[0][0]
        height = self.bottom_of_the_tank - correct_height

        return height, tank_width

    def flocs_detection(self, dst, sediment_height, tank_width, liquid_level):
        """
        Determining flocs presence

        """
        # dst - image obtained from histgram backprojection method
        dst_height, dst_width = dst.shape
        height_from_bottom = self.bottom_of_the_tank - sediment_height - dst_height
        # cropping image to SLB, eventual non-zero pixels are now only between top of liguid and SLB
        dst[height_from_bottom:] = np.zeros((-height_from_bottom, dst_width), dtype=np.uint8)
        dst = np.where(dst > 0, 1, 0)  # every pixel with even the smallest non-zero value is considered floc
        floc_sum = dst.sum()  # every floc is summed this is considered as flocs surface
        # surface of tank between SLB and top of the liquid
        surface_of_interest = (self.bottom_of_the_tank - self.top_of_the_tank - sediment_height) * tank_width
        if surface_of_interest != 0:
            # flocs surface to tank surface ratio
            floc_density = floc_sum / surface_of_interest
            # Reacting accordingly to flocs ratio
            if floc_density > 0.5 and self.print_floc_surface:
                print("Lp: " + str(self.lp) + " Detected large amount of sludge flocs (over 50% of surface above " + str(
                    sediment_height / liquid_level * 1000) + "[mL] mark) it is highly probable that sludge level readings are affected, additional operator's measurement advised.")
            elif floc_density > 0.1 and self.print_floc_surface:
                print("Lp: " + str(
                    self.lp) + " Detected significant amount of sludge flocs (beetween 10-50% of surface above " + str(
                    sediment_height / liquid_level * 1000) + "[mL] mark) which may affect sludge level readings.")
            elif floc_density > 0.005 and self.print_floc_surface:
                print("Lp: " + str(
                    self.lp) + " Detected small amount of sludge flocs (below 10% of surface above " + str(
                    sediment_height / liquid_level * 1000) + "[mL] mark).")
        else:
            floc_density = -1

        return floc_density



    def to_csv(self,data,newfilePath):
        import csv
        data = zip(data)
        timeString  = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        with open(newfilePath + timeString + ".csv", "w") as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)

    def set_up_blob_detection(self):
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 256

        params.filterByColor = True
        params.blobColor = 255

        params.filterByArea = True
        params.minArea = 10000
        params.maxArea = 9223372

        params.filterByCircularity = False
        params.filterByInertia =False
        params.filterByConvexity = False

        detector = cv2.SimpleBlobDetector_create(params)
        return params, detector

    def save_debug_image(self, img, height, percent_out,):
        height, width, channels = img.shape
        if self.img_vertical:
            img = cv2.line(img,(0,self.top_of_the_tank),(width,self.top_of_the_tank),(255,0,0),3)
            img = cv2.line(img,(0,self.bottom_of_the_tank),(width,self.bottom_of_the_tank),(255,255,0),3)
        else:
            img = cv2.line(img,(self.top_of_the_tank, 0),(self.top_of_the_tank, height),(255,0,0),3)
            img = cv2.line(img,(self.bottom_of_the_tank, 0),(self.bottom_of_the_tank, height),(255,255,0),3)
        timeString  = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        img - cv2.putText(img, str(round(percent_out, 1)) + "%", org= (int(height/8),int(width/8)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 5, color = 1000, thickness = 3)
        cv2.imwrite("./hist_out/hist_out_" + str(self.lp)+ ".jpg",img)
