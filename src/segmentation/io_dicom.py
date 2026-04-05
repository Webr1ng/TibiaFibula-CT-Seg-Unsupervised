"""
DICOM读写与掩膜封装模块
"""
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime


def load_dicom(dicom_path):
    """加载DICOM文件"""
    return pydicom.dcmread(dicom_path)


def create_dicom_mask(original_ds, mask_360, output_path):
    """
    创建DICOM格式掩膜文件，将360x360掩膜回填到512x512

    Args:
        original_ds: 原始DICOM Dataset
        mask_360: 360x360掩膜数组（胫骨=1，腓骨=2）
        output_path: 输出路径
    """
    # 回填到512x512
    full_mask_512 = np.zeros((512, 512), dtype=np.uint16)
    start = (512 - 360) // 2
    full_mask_512[start:start+360, start:start+360] = mask_360

    # 创建文件元信息
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = '1.2.3.4.5.6.7.8.9'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'

    mask_ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # 复制空间元数据
    for attr in ['PatientID', 'StudyInstanceUID', 'StudyID', 'PixelSpacing',
                 'SliceThickness', 'ImagePositionPatient', 'ImageOrientationPatient',
                 'SpacingBetweenSlices']:
        if hasattr(original_ds, attr):
            setattr(mask_ds, attr, getattr(original_ds, attr))

    mask_ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    mask_ds.Rows = 512
    mask_ds.Columns = 512
    mask_ds.PhotometricInterpretation = "MONOCHROME2"
    mask_ds.SamplesPerPixel = 1
    mask_ds.BitsAllocated = 16
    mask_ds.BitsStored = 16
    mask_ds.HighBit = 15
    mask_ds.PixelRepresentation = 0
    mask_ds.PixelData = full_mask_512.tobytes()
    mask_ds.Modality = "SEG"
    mask_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    mask_ds.SOPInstanceUID = pydicom.uid.generate_uid()
    mask_ds.SeriesDescription = "Bone Segmentation Mask"
    mask_ds.SeriesNumber = 999
    mask_ds.RescaleIntercept = 0
    mask_ds.RescaleSlope = 1

    dt = datetime.datetime.now()
    mask_ds.ContentDate = dt.strftime('%Y%m%d')
    mask_ds.ContentTime = dt.strftime('%H%M%S.%f')

    mask_ds.save_as(output_path, write_like_original=False)
