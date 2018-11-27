import tensorflow as tf
from glob import glob
from object_detection.utils import dataset_util
import numpy as np
import random
import matplotlib.pyplot as plt


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

class_names = (
    'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
    'Military', 'Commercial', 'Trains'
)

def rot(n):
  n = np.asarray(n).flatten()
  assert(n.size == 3)

  theta = np.linalg.norm(n)
  if theta:
      n /= theta
      K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

      return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
  else:
      return np.identity(3)


def get_bbox(p0, p1):
  """
  Input:
  *   p0, p1
      (3)
      Corners of a bounding box represented in the body frame.

  Output:
  *   v
      (3, 8)
      Vertices of the bounding box represented in the body frame.
  *   e
      (2, 14)
      Edges of the bounding box. The first 2 edges indicate the `front` side
      of the box.
  """
  v = np.array([
      [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
      [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
      [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
  ])
  e = np.array([
      [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
      [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
  ], dtype=np.uint8)

  return v, e

def create_tf_example(img_path,bbox,proj):
  # TODO(user): Populate the following variables from your example.
  height = 1052 # Image height
  width = 1914 # Image width
  filename = img_path # Filename of the image. Empty if image is not from file
  filename = filename.encode()
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_image_data = fid.read()
  #print(encoded_image_data)
  #encoded_image_data = 'jpg'.encode()  # Encoded image bytes
  image_format = 'jpg'.encode()  # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  '''
  img = plt.imread(img_path)

  fig1 = plt.figure(1, figsize=(16, 9))
  ax1 = fig1.add_subplot(1, 1, 1)
  ax1.imshow(img)
  ax1.axis('scaled')
  fig1.tight_layout()
  colors = ['C{:d}'.format(i) for i in range(10)]
  '''
  for k,b in enumerate(bbox):
    #if box['occluded'] is False:
    #print("adding box")
    R = rot(b[0:3])
    t = b[3:6]

    sz = b[6:9]
    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]

    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
    vert_2D = vert_2D / vert_2D[2, :]

    t = edges.T
    x = []
    y = []
    for e in t:
      #x = [vert_2D[0,2],vert_2D[0,3],vert_2D[0,6],vert_2D[0,7]]
      #y = [vert_2D[1,2],vert_2D[1,3],vert_2D[1,6],vert_2D[1,7]]
      x.append(vert_2D[0, e[0]])
      x.append(vert_2D[0, e[1]])
      y.append(vert_2D[1, e[0]])
      y.append(vert_2D[1, e[1]])
    #clr = colors[np.mod(k, len(colors))]
    
    #print ("x",x)
    
    x_min = min(x)
    x_min = max(0,x_min)
    x_min = min(1914,x_min)
    #print("x_min",x_min)
    x_max = max(x)
    x_max = max(0,x_max)
    x_max = min(1914,x_max)
    y_min = min(y)
    y_min = max(0,y_min)
    y_min = min(1052,y_min)
    y_max = max(y)
    y_max = max(0,y_max)
    y_max = min(1052,y_max)
    #ax1.plot([x_min,x_max], [y_min,y_max], color=clr)

    #ax1.plot(x, y, color=clr)
    xmins.append(float(x_min/ width))
    xmaxs.append(float(x_max / width))
    ymins.append(float(y_min/ height))
    ymaxs.append(float(y_max / height))
    #classes_text.append(class_names[int(b[9])].encode())
    classes_text.append("car".encode())
    classes.append(int(1))
  #plt.show()
  #print(xmins, xmaxs,ymins,ymaxs)
  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable
  files = glob('trainval/*/*_image.jpg')
  #random.seed(42)
  #random.shuffle(files)
  for idx in range(6600,6650):
    img_path = files[idx]
    proj = np.fromfile(img_path.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])
    try:
      bbox = np.fromfile(img_path.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
      print('[*] bbox not found.')
      bbox = np.array([], dtype=np.float32)
    bbox = bbox.reshape([-1, 11])
    tf_example = create_tf_example(img_path,bbox,proj)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
