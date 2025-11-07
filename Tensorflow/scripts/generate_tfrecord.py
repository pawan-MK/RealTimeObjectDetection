"""
Usage:
  # Create train record:
  python generate_tfrecord.py --csv_input=workspace/annotations/train_labels.csv --image_dir=workspace/images/train --output_path=workspace/annotations/train.record --label_map=workspace/annotations/label_map.pbtxt

  # Create test record:
  python generate_tfrecord.py --csv_input=workspace/annotations/test_labels.csv --image_dir=workspace/images/test --output_path=workspace/annotations/test.record --label_map=workspace/annotations/label_map.pbtxt
"""

import os
import io
import pandas as pd
import tensorflow as tf
tf.gfile =tf.io.gfile
import argparse
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Define command-line arguments
parser = argparse.ArgumentParser(description='Generate TFRecord from CSV.')
parser.add_argument('--csv_input', type=str, required=True, help='Path to the CSV input')
parser.add_argument('--image_dir', type=str, required=True, help='Path to the image directory')
parser.add_argument('--output_path', type=str, required=True, help='Path to output TFRecord')
parser.add_argument('--label_map', type=str, required=True, help='Path to label map file')
args = parser.parse_args()

def class_text_to_int(row_label, label_map_dict):
    return label_map_dict[row_label]

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map_dict))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    writer = tf.io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = pd.read_csv(args.csv_input)
    label_map_dict = label_map_util.get_label_map_dict(args.label_map)
    grouped = split(examples, 'filename')
    
    for group in grouped:
        tf_example = create_tf_example(group, path, label_map_dict)
        writer.write(tf_example.SerializeToString())
    
    writer.close()
    print(f'Successfully created the TFRecord file: {args.output_path}')

if __name__ == '__main__':
    main()