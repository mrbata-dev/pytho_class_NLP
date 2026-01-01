import os
import glob
import shutil
import pandas as pd
from xml.etree import ElementTree as et
from functools import reduce
# import os 
from shutil import move

import warnings
warnings.filterwarnings('ignore')

xmlfiles = glob.glob(r'/home/secret/Documents/pythonVirtualEnv/labels/*.xml')
# print(xmlfiles)

def extract_text(filename):
    tree = et.parse(filename)
    root = tree.getroot()

    # Extract filename, width, and height
    image_name = root.find('filename').text if root.find('filename') is not None else None
    width = root.find('size/width').text if root.find('size/width') is not None else None
    height = root.find('size/height').text if root.find('size/height') is not None else None

    parser = []
    for obj in root.findall('object'):
        name = obj.find('name').text if obj.find('name') is not None else None
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            xmin = bndbox.find('xmin').text if bndbox.find('xmin') is not None else None
            xmax = bndbox.find('xmax').text if bndbox.find('xmax') is not None else None
            ymin = bndbox.find('ymin').text if bndbox.find('ymin') is not None else None
            ymax = bndbox.find('ymax').text if bndbox.find('ymax') is not None else None
        else:
            xmin = xmax = ymin = ymax = None

        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])
    return parser

parser_all = list(map(extract_text, xmlfiles))
parser_all
# print(parser_all)

data = reduce(lambda x, y: x+y, parser_all)
# print("data:",data)
data

df = pd.DataFrame(data, columns = ['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])
# print(df['name'].value_counts())


df = df[~df['name'].str.isdigit()]
print(df['name'].value_counts())

# print(df.dtypes)
df.dtypes


#Conversion
cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
df[cols] = df[cols].astype(int)
print(df.info())

#center x, center y
df['center_x'] = ((df['xmax'] + df['xmin'])/2)/df['width']
df['center_y'] = ((df['ymax'] + df['ymin'])/2)/df['width']

df['w'] = (df['xmax'] - df['xmin'])/df['width']

df['h'] = (df['ymax'] - df['ymin'])/df['width']

print(df)


#label Endogin4
def label_encoding(x):
    labels={'book': 0, 'fan1':1, 'person':2, 'bag':3}
    try:
        return labels[x]
    except KeyError:
        return -1
df['id'] = df['name'].apply(label_encoding)
# print(df)
df

images = df['filename'].unique()
print(len(images))

#80% train and 20% test
img_df = pd.DataFrame(images, columns=['filename'])
img_train = tuple(img_df.sample(frac=0.8)['filename'])

img_test = tuple(img_df.query(f'filename not in {img_train}')['filename'])
print(img_test)

len(img_train), len(img_test)


train_df = df.query(f'filename in{img_train}')
test_df = df.query(f'filename in {img_test}')

print(train_df)

#Assign id numbet to object nae


cols=['filename', 'id', 'center_x', 'center_y', 'w', 'h']
groupby_obj_train = train_df[cols].groupby('filename')
groupby_obj_test = test_df[cols].groupby('filename')

print(groupby_obj_train)

def save_def(filename, folder_path, group_obj):
    # normalized_filename =  os.path.splitext(filename)[0].strip().lower()
    matches = glob.glob(f"data_images/{filename}.*")


    if matches:
        src_path = matches[0]
        dst_path = os.path.join(folder_path, os.path.basename(src_path))

        if os.path.exists(src_path):
            print(f"Moving {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)
        else:
            print(f"source file does not exits: {src_path}")
    else:
        print(f"matching image file not found for: {filename}")

    if filename in group_obj.groups:
        text_filename = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
        group_data = group_obj.get_group(filename).set_index('filename')
        group_data.to_csv(text_filename, sep=' ', index=False, header=False)
    else:
        print(f"Filename not found in group object: {filename}")

train_folder = 'data_images/trail_1'
test_folder = 'data_images/text_2'
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


#Applu the function
filename_series_train = pd.Series(groupby_obj_train.groups.keys())
filename_series_train.apply(lambda filename: save_def(filename, train_folder, groupby_obj_train))

filename_series_test = pd.Series(groupby_obj_test.groups.keys())
filename_series_test.apply(lambda filename: save_def(filename, train_folder, groupby_obj_test))
