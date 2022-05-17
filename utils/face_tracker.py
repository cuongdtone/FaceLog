from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

from .face_aligner import norm_crop

import cv2

class CentroidTracker():
	def __init__(self, maxDisappeared=0):

		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in self.disappeared.keys():
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects, []

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY, scrore)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()


			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				# val
				if row in usedRows or col in usedCols:
					continue
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
		# faces = sort_box(self.objects, inputCentroids, rects)
		return self.objects, inputCentroids


def find_faces(id, objects, input_centroid, faces, kpss):
	for idx, c in enumerate(input_centroid):
		if objects[id][0] == c[0] and objects[id][1] == c[1]:
			return faces[idx], kpss[idx]
	return None, None


class Track():
	def __init__(self):
		self.people_tracked = {}
		self.frame_FAS = 8
		self.frame_verify = 1
		self.verify_threshold = 0.7
		self.recog_frame = self.frame_verify

		# self.eye = Eye()
	def update(self, id, box, kps, frame, face_recognizer, employees_data, recog):

		if not id in self.people_tracked.keys():
			self.people_tracked.update({id: {'box': box,
							  'kps': kps,
							  'FAS': 1,
							  'verify': None,
							  'FAS_eye_blink':[]}}) # 0: not, 1: OK
		else:
			if box is None and kps is None:
				self.people_tracked.pop(id)
				return None
			else: # Update old person
				# self.people_tracked[id] = [box, kps]
				# FAS method
				if self.people_tracked[id]['FAS'] == 0:
					if len(self.people_tracked[id]['FAS_eye_blink']) >= self.frame_FAS:
						self.people_tracked[id]['FAS_eye_blink'] = \
							list(filter((1).__ne__, self.people_tracked[id]['FAS_eye_blink']))
						print(f'Blink detection id {id}: ', np.std(self.people_tracked[id]['FAS_eye_blink']))
						if np.std(self.people_tracked[id]['FAS_eye_blink']) >= 0.8:
							self.people_tracked[id]['FAS'] = 1 # set flag face is real
						self.people_tracked[id]['FAS_eye_blink'] = []
					else:
						pred_eye, left_eye, right_eye = self.eye.predict(frame, kps)
						self.people_tracked[id]['FAS_eye_blink'].append(pred_eye)
				# face is real
				if self.people_tracked[id]['FAS'] == 1:
					# verify is not None: pass
					if self.people_tracked[id]['verify'] is None:
						if recog:
							feet = face_recognizer.face_encoding(frame, kps)
							info = face_recognizer.face_compare(feet, employees_data)
							if info['Sim'] > self.verify_threshold:
								print(info)
								self.people_tracked[id]['verify'] = info

		return self.people_tracked[id]['verify']





