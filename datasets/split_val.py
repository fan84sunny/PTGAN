import glob
from json.tool import main
import re
import xml.dom.minidom as XD
import os.path as osp
from bases import BaseImageDataset
import os
import xml.etree.ElementTree as ET


if __name__ == "__main__":
    dataset_dir = '/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AIC21/AIC21_Track2_ReID'
    xml_dir = osp.join(dataset_dir, 'train_label.xml')
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    lable = []
    for item in root.iter('Item'):
        lable.append((item.get('imageName'), int(item.get('vehicleID')), item.get('cameraID')))
    lable_sort = sorted(lable, key = lambda s: s[1])
    print(len(lable_sort))
    d = []
    d.append([])    
    id = lable_sort[0][1]
    count = 0
    for item in lable_sort:        
        if item[1] == id:
            d[count].append(item)
        else: 
            count += 1
            d.append([])
            d[count].append(item)
            id = item[1]
            
    
    g_total = 0
    q_total = 0
    with open('g.txt', 'a+') as f:
        for data in d:
            g_num = int(len(data)*0.8)
            q_num = len(data) - g_num
            g_total += g_num
            q_total += q_num 
# '<Item imageName="000001.jpg" vehicleID="0269" cameraID="c026" />'
            for l in data[:g_num]:
                f.write('<Item imageName="%s" vehicleID="%04d" cameraID="%s" />\n' % (l[0], l[1], l[2]))

            with open('query.txt', 'a+') as f2:
                for l in data[g_num:]:
                    f2.write('<Item imageName="%s" vehicleID="%04d" cameraID="%s" />\n' % (l[0], l[1], l[2]))
