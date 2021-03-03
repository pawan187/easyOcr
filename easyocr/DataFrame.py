import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
from openpyxl import Workbook
pytesseract.pytesseract.tesseract_cmd = 'C:/Users/20315168/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

img = cv2.imread("C:/Users/20315168/Desktop/easyocr/test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

custom_config = r'-l eng --oem 1 --psm 6 '
d = pytesseract.image_to_data(thresh, config=custom_config, output_type=Output.DICT)
df = pd.DataFrame(d)
df.to_excel("Testoutput.xlsx")