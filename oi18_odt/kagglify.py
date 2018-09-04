#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:13:29 2018

@author: dzx
"""

import tensorflow as tf
import argparse
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util
from object_detection.core import data_parser
from object_detection.core import standard_fields as fields

class StringParser(data_parser.DataToNumpyParser):
  """Tensorflow Example string parser."""

  def __init__(self, field_name):
    self.field_name = field_name

  def parse(self, tf_example):
    return ''.join(list(map(bytes.decode, tf_example.features.feature[self.field_name]
                   .bytes_list.value))) if tf_example.features.feature[
                       self.field_name].HasField("bytes_list") else None

class TfExampleDetectionParser(tf_example_parser.TfExampleDetectionAndGTParser):
  """Tensorflow Example proto parser."""

  def __init__(self):
    self.items_to_handlers = {
        fields.DetectionResultFields.key:
            StringParser(fields.TfExampleFields.source_id),
        # Object detections.
        fields.DetectionResultFields.detection_boxes: (tf_example_parser.BoundingBoxParser(
            fields.TfExampleFields.detection_bbox_xmin,
            fields.TfExampleFields.detection_bbox_ymin,
            fields.TfExampleFields.detection_bbox_xmax,
            fields.TfExampleFields.detection_bbox_ymax)),
        fields.DetectionResultFields.detection_classes: (
            tf_example_parser.Int64Parser(fields.TfExampleFields.detection_class_label)),
        fields.DetectionResultFields.detection_scores: (
            tf_example_parser.FloatParser(fields.TfExampleFields.detection_score)),
    }
    self.optional_items_to_handlers = {}

def detections(d, labels):
    result = []
    for i, box in enumerate(d['detection_boxes']):
        el = [labels[d['detection_classes'][i]]['name'], d['detection_scores'][i], 
                  box[1], box[0], box[3], box[2]]
        result.append(' '.join(list(map(str, el))))
    return result

def main(parsed_args):
    label_map_path =  parsed_args.label_map #'object_detection/data/oid_bbox_trainable_label_map.pbtxt'
    input_path = parsed_args.tfr_file #'atrous_lowproposals_oid.tfrecord-00003-of-00004'
    label_map = label_map_util.load_labelmap(label_map_path)
    max_num_classes = max(item.id for item in label_map.item)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes, 
                                                                False)
    cat_index = label_map_util.create_category_index(categories)
    record_iterator = tf.python_io.tf_record_iterator(path=input_path)
    data_parser = TfExampleDetectionParser()
    processed_images = 0
    with open(parsed_args.out_file, 'w') as target:
        target.writelines('ImageId,PredictionString\n')
        for tf_rec in record_iterator:
            processed_images += 1
            example = tf.train.Example()
            example.ParseFromString(tf_rec)
            decoded_dict = data_parser.parse(example)
            target.writelines('{},{}\n'.format(decoded_dict['key'], 
                              ' '.join(detections(decoded_dict, cat_index))))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert detection TFRecord file into CSV file suitable \
                                     for Open Image detection challenge submissions on Kaggle')
    parser.add_argument('--tfr_file', required=True, 
                        help='Path to input .tfrecord-x-of-y file')
    parser.add_argument('--label_map', required=True, help='Path to label map .pbtxt  file')
    parser.add_argument('--out_file', required=True, help='Path to output file')
    args = parser.parse_args()
    main(args)


