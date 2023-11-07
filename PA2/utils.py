import matlab.engine
import numpy as np

def find_camera_pose(input):

  eng = matlab.engine.start_matlab()
  eng.addpath('functions/', nargout=0) 
  p3p = eng.PerspectiveThreePoint(input)
  eng.quit()

  camera_pose = np.array(p3p)

  return camera_pose