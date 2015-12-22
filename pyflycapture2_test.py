#!/usr/bin/python

import numpy as np
import flycapture2 as fc2
import ctypes
import sys
import time
import cv2

def getBuildInfoStr(ver):
	verStr = "FlyCapture2 library version: %d.%d.%d.%d" % \
		(ver.major, ver.minor, ver.type, ver.build)
	return verStr

def getCameraInfoStr(pCamInfo):
	camInfoStr = "\n*** CAMERA INFORMATION ***\n" \
		+ "Serial number - %u\n" \
		+ "Camera model - %s\n" \
		+ "Camera vendor - %s\n" \
		+ "Sensor - %s\n" \
		+ "Resolution - %s\n" \
		+ "Firmware version - %s\n" \
		+ "Firmware build time - %s\n\n"
	camInfoStr = camInfoStr % \
		(pCamInfo.serialNumber,
		pCamInfo.modelName,
		pCamInfo.vendorName,
		pCamInfo.sensorInfo,
		pCamInfo.sensorResolution,
		pCamInfo.firmwareVersion,
		pCamInfo.firmwareBuildTime)
	if (pCamInfo.interfaceType == fc2.FC2_INTERFACE_GIGE):
		camInfoStr = camInfoStr \
			+ "GigE major version - %u\n" \
			+ "GigE minor version - %u\n" \
			+ "User-defined name - %s\n" \
			+ "XML URL1 - %s\n" \
			+ "XML URL2 - %s\n" \
			+ "MAC address - %02X:%02X:%02X:%02X:%02X:%02X\n" \
			+ "IP address - %u.%u.%u.%u\n" \
			+ "Subnet mask - %u.%u.%u.%u\n" \
			+ "Default gateway - %u.%u.%u.%u\n"
		camInfoStr = camInfoStr % \
			(pCamInfo.gigEMajorVersion,
			pCamInfo.gigEMinorVersion,
			pCamInfo.userDefinedName,
			pCamInfo.xmlURL1,
			pCamInfo.xmlURL2,
			pCamInfo.macAddress.octets[0],
			pCamInfo.macAddress.octets[1],
			pCamInfo.macAddress.octets[2],
			pCamInfo.macAddress.octets[3],
			pCamInfo.macAddress.octets[4],
			pCamInfo.macAddress.octets[5],
			pCamInfo.ipAddress.octets[0],
			pCamInfo.ipAddress.octets[1],
			pCamInfo.ipAddress.octets[2],
			pCamInfo.ipAddress.octets[3],
			pCamInfo.subnetMask.octets[0],
			pCamInfo.subnetMask.octets[1],
			pCamInfo.subnetMask.octets[2],
			pCamInfo.subnetMask.octets[3],
			pCamInfo.defaultGateway.octets[0],
			pCamInfo.defaultGateway.octets[1],
			pCamInfo.defaultGateway.octets[2],
			pCamInfo.defaultGateway.octets[3])
	return camInfoStr

if __name__ == "__main__":
	ver = fc2.fc2Version()
	ret = fc2.fc2GetLibraryVersion(ctypes.byref(ver))
	print getBuildInfoStr(ver)

	context = fc2.fc2Context()
	camInfo = (fc2.fc2CameraInfo * 10)()
	numCamInfo = ctypes.c_uint(10)
	guid = fc2.fc2PGRGuid()

	ret = fc2.fc2CreateGigEContext(ctypes.byref(context))
	ret = fc2.fc2DiscoverGigECameras(context, camInfo, ctypes.byref(numCamInfo))
	print "Number of cameras discovered: %u" % numCamInfo.value

# 	print getCameraInfoStr(camInfo[0])

	ret = fc2.fc2Error()
	camInfo = fc2.fc2CameraInfo();
	numStreamChannels = ctypes.c_uint(0);
	imageSettingsInfo = fc2.fc2GigEImageSettingsInfo()
	rawImage = fc2.fc2Image()
	rgbImage = fc2.fc2Image()
	imageSettings = fc2.fc2GigEImageSettings()
	imageCnt = ctypes.c_int
	i = ctypes.c_uint
	filename = ctypes.c_char * 512;

	ret = fc2.fc2GetCameraFromIndex(context, 0, ctypes.byref(guid))
	if (ret != fc2.FC2_ERROR_OK):
		sys.exit("Error in fc2GetCameraFromIndex: %s\n" % fc2.fc2ErrorToDescription(ret))
		
	ret = fc2.fc2Connect(context, ctypes.byref(guid))
	if (ret != fc2.FC2_ERROR_OK):
		sys.exit("Error in fc2Connect: %s\n" % fc2.fc2ErrorToDescription(ret))
	
	ret = fc2.fc2GetGigEImageSettingsInfo(context, ctypes.byref(imageSettingsInfo))
	if (ret != fc2.FC2_ERROR_OK):
		sys.exit("Error in fc2GetGigEImageSettingsInfo: %s\n" % fc2.fc2ErrorToDescription(ret))
	
	imageSettings.offsetX = 0
	imageSettings.offsetY = 0
	imageSettings.height = imageSettingsInfo.maxHeight
	imageSettings.width = imageSettingsInfo.maxWidth
	imageSettings.pixelFormat = fc2.FC2_PIXEL_FORMAT_422YUV8
	
	ret = fc2.fc2SetGigEImageSettings(context, ctypes.byref(imageSettings))
	if (ret != fc2.FC2_ERROR_OK):
		sys.exit("Error in fc2SetGigEImageSettings: %s\n" % fc2.fc2ErrorToDescription(ret))

	ret = fc2.fc2StartCapture(context)
	if (ret != fc2.FC2_ERROR_OK):
		sys.exit("Error in fc2StartCapture: %s\n" % fc2.fc2ErrorToDescription(ret))
	
	ret = fc2.fc2CreateImage(ctypes.byref(rawImage))
	ret = fc2.fc2CreateImage(ctypes.byref(rgbImage))
	
	while True:
		ret = fc2.fc2RetrieveBuffer(context, ctypes.byref(rawImage))
		if (ret != fc2.FC2_ERROR_OK):
			print "Error in fc2RetrieveBuffer: %s\n" % fc2.fc2ErrorToDescription(ret)
			continue
	
		ret = fc2.fc2ConvertImageTo(fc2.FC2_PIXEL_FORMAT_BGR, ctypes.byref(rawImage), ctypes.byref(rgbImage))
		if (ret != fc2.FC2_ERROR_OK):
			sys.exit("Error in fc2ConvertImage: %s\n" % fc2.fc2ErrorToDescription(ret))
	
		rowBytes = rgbImage.receivedDataSize / rgbImage.rows
		
		ArrayType = ctypes.c_uint8 * rgbImage.dataSize
		addr = ctypes.addressof(rgbImage.pData.contents)
		rgbdata = np.frombuffer(ArrayType.from_address(addr),np.uint8).reshape((rgbImage.rows, rgbImage.cols, 3))

		cv2.imshow('show', rgbdata)
		if cv2.waitKey(40) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		
# 	ret = fc2.fc2SaveImage(ctypes.byref(rgbImage), "Hello.jpg", fc2.FC2_JPEG);
# 	if (ret != fc2.FC2_ERROR_OK):
# 		sys.exit("Error in fc2SaveImage: %s\n" % fc2.fc2ErrorToDescription(ret))
	
	ret = fc2.fc2StopCapture(context)
	if (ret != fc2.FC2_ERROR_OK):
		sys.exit("Error in fc2StopCapture: %s\n" % fc2.fc2ErrorToDescription(ret))

	ret = fc2.fc2Disconnect(context)
	if (ret != fc2.FC2_ERROR_OK):
		sys.exit("Error in fc2Disconnect: %s\n" % fc2.fc2ErrorToDescription(ret))



