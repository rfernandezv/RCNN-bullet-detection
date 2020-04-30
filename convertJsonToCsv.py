import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

data = pd.read_csv('file.csv',delimiter=",")
export = pd.DataFrame({'file_name':[],'x':[],'y':[],'width':[],'height':[]})

export['file_name'] = data['file_name']

i = 0

for _,row in data.iterrows():
    
    positionX = row.f.find('x":')
    positionEndX = row.f.find(',"y')
    
    positionY = row.f.find('y":')
    positionEndY = row.f.find(',"wi')
    
    positionWidth = row.f.find('dth":')
    positionEndWidth = row.f.find(',"hei')
    
    positionHeigth = row.f.find('ight":')
    positionEndHeight = row.f.find('}')
    
    x = row.f[int(positionX)+3 : int(positionEndX)]
    y = row.f[int(positionY)+3 : int(positionEndY)]
    width = row.f[int(positionWidth)+5 : int(positionEndWidth)]
    height = row.f[int(positionHeigth)+6 : int(positionEndHeight)]
    
    export['x'][i] = x
    export['y'][i] = y
    export['width'][i] = width
    export['height'][i] = height
    
    i = i + 1


export.to_csv('export.csv', index=None, sep=';')