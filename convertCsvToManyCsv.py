import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import csv

data = pd.read_csv('bullet_csv.csv',delimiter=";")
resumeData = pd.DataFrame({'image_name':[],'x1':[],'y1':[],'x2':[],'y2':[]})

resumeData['image_name'] = data['image_name']
resumeData['x1'] = data['xmin']
resumeData['x2'] = data['xmax']
resumeData['y1'] = data['ymin']
resumeData['y2'] = data['ymax']


image_name_temp = 'X'
cont = 0

for _,row in resumeData.iterrows():
    cont = cont + 1
    
    if image_name_temp != 'X':
        if image_name_temp != row.image_name:            
            
            # exportar
            export = resumeData.loc[resumeData['image_name'] == image_name_temp]
            positionExt = image_name_temp.find('.JPG')
            name = image_name_temp[0:positionExt]            
            nameCsv = name + '.csv'
            
            with open(nameCsv, 'w', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=' ')
                filewriter.writerow([str(cont)] )
                
                for _,rowExport in export.iterrows():
                    print(nameCsv)
                    filewriter.writerow([str(int(rowExport.x1)),str(int(rowExport.y1)),str(int(rowExport.x2)),str(int(rowExport.y2))])
            ####
            
            # limpiar valores #
            #print("image_name: "+image_name_temp+", cont: "+str(cont))
            cont = 0
            ###
       
    image_name_temp = row.image_name
    
 # exportar
export = resumeData.loc[resumeData['image_name'] == image_name_temp]
positionExt = image_name_temp.find('.JPG')
name = image_name_temp[0:positionExt]            
nameCsv = name + '.csv'

with open(nameCsv, 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
    filewriter.writerow([str(cont)] )
    
    for _,rowExport in export.iterrows():
        filewriter.writerow([str(rowExport.x1),str(rowExport.y1),str(rowExport.x2),str(rowExport.y2)])
####

