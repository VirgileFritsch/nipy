"""
Utilities for extracting masks from EPI images and applying them to time
series.
"""

# Major scientific libraries imports
import numpy as np
# Neuroimaging libraries imports
from nipy.io.imageformats import load, nifti1, save, AnalyzeImage


################################################################################
# Utilities to calculate masks
################################################################################

def largest_cc(mask):
    """ Return the largest connected component of a 3D mask array.

        Parameters
        -----------
        mask: 3D boolean array
            3D array indicating a mask.
        
        Returns
        --------
        mask: 3D boolean array 
            3D array indicating a mask, with only one connected component.    
    """
    # Late import of scipy
    from scipy import ndimage

    # We use asarray to be able to work with masked arrays.
    mask = np.asarray(mask)
    labels, label_nb = ndimage.label(mask)
    if not label_nb:
        raise ValueError('No non-zero values: no connect components')
    return labels ==  np.bincount(labels.flat)[1:].argmax() + 1


# FIXME: Should this function be replaced by the native functionality
# added to brifti
def get_unscaled_img(fname):
    ''' Function to get image, data without scalefactor applied

    If the image is of Analyze type, and is integer format, and has
    single scalefactor that is usually applied, then read the raw
    integer data from disk, rather than using the higher-level get_data
    method, that would apply the scalefactor.  We do this because there
    seemed to be images for which the integer binning in the raw file
    data was needed for the histogram-like mask calculation in
    ``compute_mask_files``.

    By loading the image in this function we can guarantee that the
    image as loaded from disk is the source of the current image data.

    Parameters
    ----------
    fname : str
       filename of image

    Returns
    -------
    img : imageformats Image object
    arr : ndarray
    '''
    img = load(fname)
    if isinstance(img, AnalyzeImage):
        dt = img.get_data_dtype()
        if dt.kind in ('i', 'u'):
            from nipy.io.imageformats.header_ufuncs import read_unscaled_data
            from nipy.io.imageformats.volumeutils import allopen
            # get where the image data is, given input filename
            ft = img.filespec_to_files(fname)
            hdr = img.get_header()
            # read unscaled data from disk
            return img, read_unscaled_data(hdr, allopen(ft['image']))
    return img, img.get_data()


def compute_mask_files(input_filename, output_filename=None, 
                        return_mean=False, m=0.2, M=0.9, cc=1):
    """
    Compute a mask file from fMRI nifti file(s)

    Compute and write the mask of an image based on the grey level
    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    m and M of the total image histogram.

    In case of failure, it is usually advisable to increase m.
   
    Parameters
    ----------
    input_filename : string
        nifti filename (4D) or list of filenames (3D).
    output_filename : string or None, optional
        path to save the output nifti image (if not None).
    return_mean : boolean, optional
        if True, and output_filename is None, return the mean image also, as 
        a 3D array (2nd return argument).
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.

    Returns
    -------
    mask : 3D boolean array 
        The brain mask
    mean_image : 3d ndarray, optional
        The main of all the images used to estimate the mask. Only
        provided if `return_mean` is True.
    """
    if isinstance(input_filename, basestring):
        # One single filename
        nim, vol_arr = get_unscaled_img(input_filename)
        header = nim.get_header()
        affine = nim.get_affine()
        # Make a copy, to avoid holding a reference on the full array,
        # and thus polluting the memory.
        if vol_arr.ndim == 4:
            mean_volume = vol_arr.mean(axis=-1)
            first_volume = vol_arr[:,:,:,0].copy()
        elif vol_arr.ndim == 3:
            mean_volume = first_volume = vol_arr
        else:
            raise ValueError('Need 4D file for mask')
        del vol_arr
    else:
        # List of filenames
        if len(input_filename) == 0:
            raise ValueError('input_filename should be a non-empty '
                'list of file names')
        # We have several images, we do mean on the fly, 
        # to avoid loading all the data in the memory
        # We do not use the unscaled data here?:
        # if the scalefactor is being used to record real
        # differences in intensity over the run this would break
        for index, filename in enumerate(input_filename):
            nim = load(filename)
            if index == 0:
                first_volume = nim.get_data().squeeze()
                mean_volume = first_volume.copy().astype(np.float32)
                header = nim.get_header()
                affine = nim.get_affine()
            else:
                mean_volume += nim.get_data().squeeze()
        mean_volume /= float(len(input_filename))
    del nim

    mask = compute_mask(mean_volume, first_volume, m, M, cc)
      
    if output_filename is not None:
        header['descrip'] = 'mask'
        output_image = nifti1.Nifti1Image(mask.astype(np.uint8), 
                                            affine=affine, 
                                            header=header)
        save(output_image, output_filename)
    if not return_mean:
        return mask
    else:
        return mask, mean_volume


def compute_mask(mean_volume, reference_volume=None, m=0.2, M=0.9, 
                                                cc=1):
    """
    Compute a mask file from fMRI data in 3D or 4D ndarrays.

    Compute and write the mask of an image based on the grey level
    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    m and M of the total image histogram.

    In case of failure, it is usually advisable to increase m.
   
    Parameters
    ----------
    mean_volume : 3D ndarray 
        mean EPI image, used to compute the threshold for the mask.
    reference_volume: 3D ndarray, optional
        reference volume used to compute the mask. If none is give, the 
        mean volume is used.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.

    Returns
    -------
    mask : 3D boolean ndarray 
        The brain mask
    """
    if reference_volume is None:
        reference_volume = mean_volume
    inputVector = np.sort(mean_volume.reshape(-1))
    limiteinf = np.floor(m * len(inputVector))
    limitesup = np.floor(M * len(inputVector))#inputVector.argmax())

    delta = inputVector[limiteinf + 1:limitesup + 1] \
            - inputVector[limiteinf:limitesup]
    ia = delta.argmax()
    threshold = 0.5 * (inputVector[ia + limiteinf] 
                        + inputVector[ia + limiteinf  +1])
    
    mask = (reference_volume >= threshold)

    if cc:
        mask = largest_cc(mask)
    return mask.astype(bool)


def compute_mask_sessions(session_files, m=0.2, M=0.9, cc=1, threshold=0.5):
    """ Compute a common mask for several sessions of fMRI data.

        Uses the mask-finding algorithmes to extract masks for each
        session, and then keep only the main connected component of the
        a given fraction of the intersection of all the masks.

 
    Parameters
    ----------
    session_files : list of list of strings
        A list of list of nifti filenames. Each inner list
        represents a session.
    threshold : float, optional
        the inter-session threshold: the fraction of the
        total number of session in for which a voxel must be in the
        mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.

    Returns
    -------
    mask : 3D boolean ndarray 
        The brain mask
    """
    mask = None
    for session in session_files:
        this_mask = compute_mask_files(session,
                                       m=m, M=M,
                                       cc=cc).astype(np.int8)
        if mask is None:
            mask = this_mask
        else:
            mask += this_mask
        # Free memory early
        del this_mask
        
    # Take the "half-intersection", i.e. all the voxels that fall within
    # 50% of the individual masks.
    mask = (mask > threshold*len(session_files))
   
    if cc:
        # Select the largest connected component (each mask is
        # connect, but the half-interesection may not be):
        mask = largest_cc(mask)

    return mask.astype(np.bool)


def intersect_masks(input_masks, output_filename=None, 
                                        threshold=0.5, cc=True):
    """
    Given a list of input mask images, generate the output image which
    is the the threshold-level intersection of the inputs 

    
    Parameters
    ----------
    input_masks: list of strings or ndarrays
        paths of the input images nsubj set as len(input_mask_files), or
        individual masks.
    output_filename, string:
        Path of the output image, if None no file is saved.
    threshold: float within [0, 1], optional
        gives the level of the intersection.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    cc: bool, optional
        If true, extract the main connected component
        
    Returns
    -------
    grp_mask, boolean array of shape the image shape
    """  
    grp_mask = None 

    for this_mask in input_masks:
        if isinstance(this_mask, basestring):
            # We have a filename
            this_mask = load(this_mask).get_data()
        if grp_mask is None:
            grp_mask = this_mask.copy().astype(np.int)
        else:
            grp_mask += this_mask
    
    grp_mask = grp_mask>(threshold*len(input_masks))
    if np.any(grp_mask>0) and cc:
        grp_mask = largest_cc(grp_mask)
    
    if output_filename is not None:
        if isinstance(input_masks[0], basestring):
            nim = load(input_masks[0]) 
            header = nim.get_header()
            affine = nim.get_affine()
        else:
            header = dict()
            affine = np.eye(4)
        header['descrip'] = 'mask image'
        output_image = nifti1.Nifti1Image(grp_mask.astype(np.uint8),
                                            affine=affine,
                                            header=header,
                                         )
        output_image.save(output_filename)

    return grp_mask>0


################################################################################
# Time series extraction
################################################################################

# FIXME: This function should probably get a 'single_session' flag to work 
# without any surprises on single session situations.
def series_from_mask(session_files, mask, dtype=np.float32,
                squeeze=False, smooth=False):
    """ Read the time series from the given sessions filenames, using the mask.

        Parameters
        -----------
        session_files: list of list of nifti file names. 
            Files are grouped by session.
        mask: 3d ndarray
            3D mask array: true where a voxel should be used.
        squeeze: boolean, optional
            If squeeze is True, the data array is squeezed before return.
        smooth: False or float, optional
            If smooth is not False, it gives the size, in voxel of the
            spatial smoothing to apply to the signal.
        
        Returns
        --------
        session_series: ndarray
            3D array of time course: (session, voxel, time)
        header: header object
            The header of the first file.
    """
    # XXX: What if the file lengths do not match!
    mask = mask.astype(np.bool)
    nb_time_points = len(session_files[0])
    if len(session_files[0]) == 1:
        # We have a 4D nifti file
        nb_time_points = load(session_files[0][0]).get_data().shape[-1]
    session_series = np.zeros((len(session_files), mask.sum(),
                                            nb_time_points),
                                    dtype=dtype)

    for session_index, filenames in enumerate(session_files):
        if len(filenames) == 1:
            # We have a 4D nifti file
            data_file = load(filenames[0])
            data = data_file.get_data()
            if not 'header' in locals():
                header = data_file.get_header()
            if smooth:
                affine = data_file.get_affine()[:3, :3]
                smooth_sigma = np.dot(affine, np.ones(3))*smooth
                from scipy import ndimage
                data = np.asarray(data) # Get rid of memmapping
                for this_data in np.rollaxis(data, -1):
                    this_data[:] = ndimage.gaussian_filter(this_data,
                                                           smooth_sigma)
            session_series[session_index, :, :] = data[mask].astype(dtype)
            # Free memory early
            del data, data_file
        else:
            for file_index, filename in enumerate(filenames):
                data_file = load(filename)
                data = data_file.get_data()
                if smooth:
                    affine = data_file.get_affine()[:3, :3]
                    smooth_sigma = np.dot(affine, np.ones(3))*smooth
                    from scipy import ndimage
                    data = ndimage.gaussian_filter(data, smooth_sigma)
                    
                session_series[session_index, :, file_index] = \
                                data[mask].astype(np.float32)
                # Free memory early
                if not 'header' in locals():
                    header = data_file.get_header()
                del data

    if squeeze:
        session_series = session_series.squeeze()
    return session_series, header


