# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import scipy.sparse as sps

from nibabel import load, Nifti1Image

from . import discrete_domain as ddom


##############################################################################
# class MultipleROI
##############################################################################
class MultipleROI(object):
    """
    This is a class to represent multiple ROI objects, where the
    reference to a given domain is explicit.

    A mutliple ROI object is a set of ROI defined on a given domain,
    each having its own 'region-level' characteristics (ROI features).

    Every voxel of the domain can have its own characteristics yet,
    defined at the 'voxel-level', but those features can only be accessed
    familywise (i.e. the values are grouped by ROI).

    Parameters
    ----------
    `k`: int,
      Number of ROI in the MultipleROI object
    `label`: array of shape (domain.size), dtype=np.int,
      An array use to define which voxel belongs to which ROI.
      If label.shape == (domain.size, ), the label values greater
      than -1 correspond to subregions labelling.
      The labels should not be accessed outside this class. One has to
      use the API mapping methods instead.
    `features`: dict{str: list of object, length=self.k}
      Describe the voxels features, grouped by ROI
    `roi_features`: dict{str: array-like, shape=(self.k, roi_feature_dim)
      Describe the ROI features. A special feature, `id`, is read-only and
      is used to give an unique identifier for region, which is persistent
      through the MROI objects manipulations. On should access the different
      ROI's features using ids.

    """

    def __init__(self, domain, label, id=None):
        """Initialize MultipleROI instance

        Parameters
        ----------
        domain: ROI instance
          defines the spatial context of the MultipleROI
        label: array of shape (domain.size), dtype=np.int,
               or array of shape (n_roi, domain.size), dtype=np.int
          An array use to define which voxel belongs to which ROI.
          If label.shape == (domain.size, ), the label values greater
          than -1 correspond to subregions labelling.
          The labels should not be accessed outside this class. One has to
          use the select_id() mapping method instead.
        id: array of shape (n_roi)
          Define the ROI identifiers. Once an id has been associated to a ROI
          it becomes impossible to change it using the API. Hence, one
          should access ROI through their id to avoid hazardous manipulations.

        """
        if not isinstance(label, sps.coo.coo_matrix):
            label = np.asarray(label)

        if label.size == 0:
            if isinstance(label, sps.coo.coo_matrix):
                self.voxels_to_rois_map = sps.coo_matrix(
                    ([], ([], [])), shape=label.shape)
            else:
                self.voxels_to_rois_map = sps.coo_matrix(
                    ([], ([], [])), shape=(1, domain.size))
        elif label.ndim == 2 and label.shape[1] == domain.size:
            if label.shape[1] != domain.size:
                raise ValueError("inconsistent dimensions of labels")
            if isinstance(label, sps.coo.coo_matrix):
                self.voxels_to_rois_map = label
            else:
                self.voxels_to_rois_map = sps.coo_matrix(
                    (np.ones(np.sum(label > 0)), np.where(label > 0)),
                    shape=label.shape)
        else:
            # check that label shape is consistent with domain
            if np.size(label) != domain.size:
                raise ValueError("inconsistent size of labels")
            label = np.reshape(label, label.size).astype(np.int)
            # redefine labels so that they are consecutive integers
            lmap = np.unique(label[label > - 1])
            for i, k in enumerate(lmap):
                label[label == k] = i
            # convert labels to a matrix
            if np.max(label) > -1:
                self.voxels_to_rois_map = sps.coo_matrix(
                    (np.ones(np.sum(label > -1)),
                     (label[label > -1], np.where(label > -1)[0])),
                    shape=(np.max(label) + 1, label.size))
            else:
                self.voxels_to_rois_map = sps.coo_matrix(
                    ([], ([], [])), shape=(1, label.size))
        self.domain = domain
        self.update_roi_number()

        # initialize empty feature/roi_feature dictionaries
        self.features = {}
        self.roi_features = {}
        # set `id` feature: unique and persistent identifier for each roi
        if id is None:
            # ids correspond to initial labels
            self.set_roi_feature('id', np.arange(self.k))
        else:
            # use user-provided ids
            if len(id) != self.k:
                raise ValueError("incorrect shape for `id`")
            self.set_roi_feature('id', id)

    ###
    # Methods for internal use: id and labels manipulations
    ###
    def update_roi_number(self):
        """Update number of regions according to labels

        Labels are used as a map to associate voxels to a given ROI.
        It is an inner object that should not be accessed outside this class.
        The number of nodes is updated appropriately.

        Note
        ----
        This method must be called everytime the MROI structure is modified.

        """
        if self.voxels_to_rois_map.size > 0:
            valid_rows = np.where(
                self.voxels_to_rois_map.tocsr().sum(1) > 0.)[0]
            valid_mask = np.asarray(
                [(i in valid_rows) for i in self.voxels_to_rois_map.row])
            new_data = np.ones(valid_mask.sum())
            new_row = self.voxels_to_rois_map.row[valid_mask]
            for i, k in enumerate(np.sort(np.unique(new_row))):
                new_row[new_row == k] = i
            new_col = self.voxels_to_rois_map.col[valid_mask]
            self.voxels_to_rois_map = sps.coo_matrix(
                (new_data, (new_row, new_col)),
                shape=(valid_rows.size, self.voxels_to_rois_map.shape[1]))
            # update number of ROIs
            self.k = self.voxels_to_rois_map.shape[0]
        else:
            self.k = 0

    def get_ids(self):
        """Return ROI's id list.

        Users must access ROIs with the use of the identifiers of this list
        and the methods that give access to their properties/features.

        """
        return self.get_roi_feature('id')

    def select_id(self, id, roi=True):
        """Convert a ROI id into an index to be used to index features safely.

        Parameter
        ---------
        id: any hashable type, must be in self.get_ids()
          The id of the region one wants to access.
        roi: boolean
          If True (default), return the ROI index in the ROI list.
          If False, return the indices of the voxels of the ROI with the given
          id. That way, internal access to self.voxels_to_rois_map can be made.

        Return
        ------
        index: int or np.array of shape (roi.size, )
          Either the position of the ROI in the ROI list (if roi == True),
          or the positions of the voxels of the ROI with id `id`
          with respect to the self.voxels_to_rois_map array.

        """
        if id not in self.get_ids():
            raise ValueError("Unexisting `id` provided")
        if roi:
            index = int(np.where(self.get_ids() == id)[0])
        else:
            index = self.voxels_to_rois_map.tocsr()[
                self.select_id(id)].nonzero()[1]
        return index

    ###
    # General purpose methods
    ###
    def copy(self):
        """Returns a copy of self.

        Note that self.domain is not copied.

        """
        cp = MultipleROI(
            self.domain, self.voxels_to_rois_map.copy(), id=self.get_ids())
        for fid in self.features.keys():
            f = self.get_feature(fid)
            sf = [np.array(f[k]).copy() for k in range(self.k)]
            cp.set_feature(fid, sf)
        for fid in self.roi_features.keys():
            cp.set_roi_feature(fid, self.get_roi_feature(fid).copy())
        return cp

    ###
    # Getters for very basic features or roi features
    ###
    def get_coord(self, id=None):
        """Get coordinates of ROI's voxels

        Parameters
        ----------
        id: any hashable type
          Id of the ROI from which we want the voxels' coordinates.
          Can be None (default) if we want all ROIs's voxels coordinates.

        Return
        ------
        coords: array-like, shape=(roi_size, domain_dimension),
          if an id is provided,
             or list of arrays of shape(roi_size, domain_dimension)
          if no id provided (default)

        """
        if id is not None:
            coords = self.domain.coord[self.select_id(id, roi=False)]
        else:
            coords = [self.domain.coord[self.select_id(k, roi=False)]
                      for k in self.get_ids()]
        return coords

    def get_size(self, id=None):
        """Get ROI size (counted in terms of voxels)

        Parameters
        ----------
        id: any hashable type
          Id of the ROI from which we want to get the size.
          Can be None (default) if we want all ROIs's sizes.

        Return
        ------
        size: int
          if an id is provided,
             or list of int
          if no id provided (default)

        """
        if id is not None:
            size = np.size(self.select_id(id, roi=False))
        else:
            size = np.array(
                [np.size(self.select_id(k, roi=False))
                 for k in self.get_ids()])
        return size

    def get_local_volume(self, id=None):
        """Get volume of ROI's voxels

        Parameters
        ----------
        id: any hashable type
          Id of the ROI from which we want the voxels' volumes.
          Can be None (default) if we want all ROIs's voxels volumes.

        Return
        ------
        loc_volume: array-like, shape=(roi_size, ),
          if an id is provided,
             or list of arrays of shape(roi_size, )
          if no id provided (default)

        """
        if id is not None:
            loc_volume = self.domain.local_volume[
                self.select_id(id, roi=False)]
        else:
            loc_volume = [self.domain.local_volume[
                    self.select_id(k, roi=False)] for k in self.get_ids()]
        return loc_volume

    def get_volume(self, id=None):
        """Get ROI volume

        Parameters
        ----------
        id: any hashable type
          Id of the ROI from which we want to get the volume.
          Can be None (default) if we want all ROIs's volumes.

        Return
        ------
        volume: float
          if an id is provided,
             or list of float
          if no id provided (default)

        """
        if id is not None:
            volume = np.sum(self.get_local_volume(id))
        else:
            volume = np.asarray([np.sum(k) for k in self.get_local_volume()])
        return volume

    ###
    # Methods for features manipulation (user level)
    ###
    def get_feature(self, fid, id=None):
        """Return a voxel-wise feature, grouped by ROI.

        Parameters
        ----------
        fid: str,
          Feature to be returned
        id: any hashable type
          Id of the ROI from which we want to get the feature.
          Can be None (default) if we want all ROIs's features.

        Return
        ------
        feature: array-like, shape=(roi_size, feature_dim)
          if an id is provided,
             or list of arrays, shape=(roi_size, feature_dim)
          if no id provided (default)

        """
        if fid not in self.features:
            raise ValueError("the `%s` feature does not exist" % fid)
        if id is not None:
            feature = np.asarray(self.features[fid][self.select_id(id)])
        else:
            feature = self.features[fid]
        return feature

    def set_feature(self, fid, data, id=None, override=False):
        """Append or modify a feature

        Parameters
        ----------
        fid: str,
          feature identifier
        data: list of self.k arrays of shape(self.size[k], p) or
              array of shape(self.size[k])
          the feature data
        id: any hashable type
          Id of the ROI from which we want to set the feature.
          Can be None (default) if we want to set all ROIs's features.
        override: bool, optional,
                  Allow feature overriding

        Note that we cannot create a feature having the same name than
        a ROI feature.

        """
        # ensure that the `id` field will not be modified
        if fid == 'id':
            override = False
        # check the feature is already present if setting a single roi
        if fid not in self.features and len(data) != self.k:
            raise ValueError("`%s` feature does not exist, create it first"
                             % fid)
        if fid in self.roi_features:
            raise ValueError("a roi_feature called `%s` already exists" % fid)
        # check we will not override anything
        if fid in self.features and not override:
            #TODO: raise a warning
            return
        # modify one particular region
        if id is not None:
            # check data size
            roi_size = self.get_size(id)
            if len(data) != roi_size:
                raise ValueError("data for region `%i` should have length %i"
                                 % (id, roi_size))
            # update feature
            the_feature = self.get_feature(fid, id)
            the_feature[self.select_id(id)] = data
        # modify all regions
        else:
            # check data size
            if len(data) != self.k:
                raise ValueError("data should have length %i" % self.k)
            for k in self.get_ids():
                if len(data[self.select_id(k)]) != self.get_size(k):
                    raise ValueError('Wrong data size for region `%i`' % k)
            self.features.update({fid: data})

    def representative_feature(self, fid, method='mean', id=None,
                               assess_quality=False):
        """Compute a ROI representative of a given feature.

        Parameters
        ----------
        fid: str,
          Feature id
        method: str,
          Method used to compute a representative.
          Chosen among 'mean' (default), 'max', 'median', 'min',
          'weighted mean'.
        id: any hashable type
          Id of the ROI from which we want to extract a representative feature.
          Can be None (default) if we want to get all ROIs's representatives.
        assess_quality: bool
          If True, a new roi feature is created, which represent the quality
          of the feature representative (the number of non-nan value for the
          feature over the ROI size).
          Default is False.

        Return
        ------
        summary_feature: np.ndarray, shape=(self.k, feature_dim)
          Representative feature computed according to `method`.

        """
        rf = []
        eps = 1.e-15
        feature_quality = np.zeros(self.k)
        for i, k in enumerate(self.get_ids()):
            f = self.get_feature(fid, k)
            # NaN-resistant representative
            if f.ndim == 2:
                nan = np.isnan(f.sum(1))
            else:
                nan = np.isnan(f)
            # feature quality
            feature_quality[i] = (~nan).sum() / float(nan.size)
            # compute representative
            if method == "mean":
                rf.append(np.mean(f[~nan], 0))
            if method == "weighted mean":
                lvk = self.get_local_volume(k)[~nan]
                tmp = np.dot(lvk, f[~nan].reshape((-1, 1))) / \
                    np.maximum(eps, np.sum(lvk))
                rf.append(tmp)
            if method == "min":
                rf.append(np.min(f[~nan]))
            if method == "max":
                rf.append(np.max(f[~nan]))
            if method == "median":
                rf.append(np.median(f[~nan], 0))
        if id is not None:
            summary_feature = rf[self.select_id(id)]
        else:
            summary_feature = rf

        if assess_quality:
            self.set_roi_feature('%s_quality' % fid, feature_quality)
        return np.array(summary_feature)

    def remove_feature(self, fid):
        """Remove a certain feature

        Parameters
        ----------
        fid: str,
          Feature id

        Return
        ------
        The removed feature.

        """
        return self.features.pop(fid)

    def feature_to_voxel_map(self, fid, roi=False, method="mean"):
        """Convert a feature to a flat voxel-mapping array.

        Paramters
        ---------
        fid: str,
          Identifier of the feature to be mapped.
        roi: bool,
          If true, compute the map from a ROI feature.
        method: str,
          Representative feature computation method if `fid` is a feature
          and `roi` is True.

        Return
        ------
        res: array-like, shape=(domain.size, feature_dim)
          A flat array, giving the correspondence between voxels
          and the feature.

        """
        res = np.zeros(self.voxels_to_rois_map.shape[1])
        if not roi:
            f = self.get_feature(fid)
            for id in self.get_ids():
                res[self.select_id(id, roi=False)] = f[self.select_id(id)]
        else:
            if fid in self.roi_features.keys():
                f = self.get_roi_feature(fid)
                for id in self.get_ids():
                    res[self.select_id(id, roi=False)] = f[self.select_id(id)]
            elif fid in self.features.keys():
                f = self.representative_feature(fid, method=method)
                for id in self.get_ids():
                    res[self.select_id(id, roi=False)] = f[self.select_id(id)]
            else:
                raise ValueError("Wrong feature id provided")
        return res

    def integrate(self, fid=None, id=None):
        """Integrate  certain feature on each ROI and return the k results

        Parameters
        ----------
        fid : str
          Feature identifier.
          By default, the 1 function is integrated, yielding ROI volumes.
        id: any hashable type
          The ROI on which we want to integrate.
          Can be None if we want the results for every region.

        Returns
        -------
        lsum = array of shape (self.k, self.feature[fid].shape[1]),
          The results

        """
        if fid == None:
            # integrate the 1 function if no feature id provided
            if id is not None:
                lsum = self.get_volume(id)
            else:
                lsum = [self.get_volume(k) for k in self.get_ids()]
        else:
            if id is not None:
                slvk = np.expand_dims(self.get_local_volume(id), 1)
                sfk = self.get_feature(fid, id)
                sfk = np.reshape(sfk, (-1, 1))
                lsum = np.sum(sfk * slvk, 0)
            else:
                lsum = []
                for k in self.get_ids():
                    slvk = np.expand_dims(self.get_local_volume(k), 1)
                    sfk = self.get_feature(fid, k)
                    sfk = np.reshape(sfk, (-1, 1))
                    sumk = np.sum(sfk * slvk, 0)
                    lsum.append(sumk)
        return np.array(lsum)

    def plot_feature(self, fid, ax=None):
        """Boxplot the distribution of features within ROIs.
        Note that this assumes 1-d features.

        Parameters
        ----------
        fid: string
             the feature identifier
        ax: axis handle, optional

        """
        f = self.get_feature(fid)
        if ax is None:
            import matplotlib.pylab as mp
            mp.figure()
            ax = mp.subplot(111)
        ax.boxplot(f)
        ax.set_title('ROI-level distribution for feature %s' % fid)
        ax.set_xlabel('Region index')
        ax.set_xticks(np.arange(1, self.k + 1))
        return ax

    ###
    # Methods for ROI features manipulation (user level)
    ###
    def get_roi_feature(self, fid, id=None):
        """
        """
        if id is not None:
            feature = self.roi_features[fid][self.select_id(id)]
        else:
            feature = np.asarray(self.roi_features[fid])
        return feature

    def set_roi_feature(self, fid, data, id=None, override=False):
        """Append or modify a ROI feature

        Parameters
        ----------
        fid: str,
          feature identifier
        data: list of self.k features or a single feature
          The ROI feature data
        id: any hashable type
          Id of the ROI of which we want to set the ROI feature.
          Can be None (default) if we want to set all ROIs's ROI features.
        override: bool, optional,
                  Allow feature overriding

        Note that we cannot create a ROI feature having the same name than
        a feature.
        Note that the `id` feature cannot be modified as an internal
        component.

        """
        # check we do not modify the `id` feature
        if 'id' in self.roi_features.keys() and fid == 'id':
            return
        # check we will not override anything
        if fid in self.roi_features and not override:
            #TODO: raise a warning
            return
        # check the feature is already present if setting a single roi
        if fid not in self.roi_features and len(data) != self.k:
            raise ValueError("`%s` feature does not exist, create it first")
        if fid in self.features:
            raise ValueError("a feature called `%s` already exists" % fid)

        # modify one particular region
        if id is not None:
            # check data size
            if len(data) != 1:
                raise ValueError("data for region `%i` should have length 1")
            # update feature
            the_feature = self.get_roi_feature(fid)
            the_feature[self.select_id(id)] = data
        else:
            # check data size
            if len(data) != self.k:
                raise ValueError("data should have length %i" % self.k)
            self.roi_features.update({fid: np.ravel(data)})

    def remove_roi_feature(self, fid):
        """Remove a certain ROI feature.

        The `id` ROI feature cannot be removed.

        Return
        ------
        The removed Roi feature.

        """
        if fid != 'id':
            feature = self.roi_features.pop(fid)
        else:
            feature = self.get_ids()
        return feature
        #TODO: raise a warning otherwise

    def to_image(self, fid=None, roi=False, method="mean", descrip=None):
        """Generates a label image that represents self.

        Parameters
        ----------
        fid: str,
          Feature to be represented. If None, a binary image of the MROI
          domain will be we created.
        roi: bool,
          Whether or not to write the desired feature as a ROI one.
          (i.e. a ROI feature corresponding to `fid` will be looked upon,
          and if not found, a representative feature will be computed from
          the `fid` feature).
        method: str,
          If a feature is written as a ROI feature, this keyword tweaks
          the way the representative feature is computed.
        descrip: str,
          Description of the image, to be written in its header.

        Note
        ----
        requires that self.dom is an ddom.NDGridDomain

        Return
        ------
        nim: nibabel nifti image
          Nifti image corresponding to the ROI feature to be written.

        """
        if not isinstance(self.domain, ddom.NDGridDomain):
            print 'self.domain is not an NDGridDomain; nothing was written.'
            return None

        if fid is None:
            # write a binary representation of the domain if no fid provided
            nim = self.domain.to_image(
                data=(self.voxels_to_rois_map.tocsc().sum(0) > 0).astype(int))
            if descrip is None:
                descrip = 'binary representation of MROI'
        else:
            data = -np.ones(self.voxels_to_rois_map.shape[1])
            tmp_image = self.domain.to_image()
            mask = tmp_image.get_data().copy().astype(bool)
            if not roi:
                # write a feature
                if fid not in self.features:
                    raise ValueError("`%s` feature could not be found" % fid)
                for i in self.get_ids():
                    data[self.select_id(i, roi=False)] = \
                        self.get_feature(fid, i)
            else:
                # write a roi feature
                if fid in self.roi_features:
                    # write from existing roi feature
                    for i in self.get_ids():
                        data[self.select_id(i, roi=False)] = \
                            self.get_roi_feature(
                            fid, i)
                elif fid in self.features:
                    # write from representative feature
                    summary_feature = self.representative_feature(
                        fid, method=method)
                    for i in self.get_ids():
                        data[self.select_id(i, roi=False)] = \
                            summary_feature[self.select_id(i)]
            # MROI object was defined on a masked image: we square it back.
            wdata = -np.ones(mask.shape, data.dtype)
            wdata[mask] = data
            nim = Nifti1Image(wdata, tmp_image.get_affine())
        # set description of the image
        if descrip is not None:
            nim.get_header()['descrip'] = descrip
        return nim

    ###
    # ROIs structure manipulation
    ###
    def select_roi(self, id_list):
        """Returns an instance of MROI with only the subset of chosen ROIs.

        Parameters
        ----------
        id_list: list of id (any hashable type)
          The id of the ROI to be kept in the structure.

        """
        # handle the case of an empty selection
        if len(id_list) == 0:
            self = MultipleROI(
                self.domain, -np.ones(self.voxels_to_rois_map.shape[1]))
            return
        # convert id to indices
        id_list_pos = np.ravel([self.select_id(k) for k in id_list])
        # set new labels (= map between voxels and ROI)
        for id in self.get_ids():
            if id not in id_list:
                self.voxels_to_rois_map.data[
                    self.voxels_to_rois_map.row == self.select_id(id)] = 0.
        self.update_roi_number()
        self.roi_features['id'] = np.ravel([id_list])

        # set new features
        # (it's ok to do that after labels and id modification since we are
        # poping out the former features and use the former id indices)
        for fid in self.features.keys():
            f = self.remove_feature(fid)
            sf = [f[id] for id in id_list_pos]
            self.set_feature(fid, sf)
        # set new ROI features
        # (it's ok to do that after labels and id modification since we are
        # poping out the former features and use the former id indices)
        for fid in self.roi_features.keys():
            if fid != 'id':
                f = self.remove_roi_feature(fid)
                sf = np.ravel(f[id_list_pos])
                self.set_roi_feature(fid, sf)


def mroi_from_array(label_image, affine=None, nn=0, threshold=-1):
    """Return a MultipleROI from an n-d int array

    Parameters
    ----------
    label_image: np.array instance
      An array that yields the regions.
    affine: np.array, optional
      Affine transform that maps the array coordinates
      to some embedding space by default, this is np.eye(dim+1, dim+1).
    nn: int,
      Neighboring system considered.
      Unused at the moment.
    threshold: float,
      Threshold on labels to define the domain.

    Note
    ----
    Only labels > `threshold` are considered.

    """
    mask = label_image > threshold
    if mask.sum() == 0:
        # if no label above the threshold, the subdomain becomes the whole img
        dom = ddom.grid_domain_from_binary_array(
            ~mask, affine=affine, nn=nn)
    else:
        dom = ddom.grid_domain_from_binary_array(
            mask, affine=affine, nn=nn)
    return MultipleROI(dom, label_image[mask].astype(np.int))


def mroi_from_image(mim, nn=18, threshold=-1):
    """Return a MultipleROI instance from the input mask image.

    Parameters
    ----------
    mim: NiftiIImage instance, or string path toward such an image
      supposedly a label image
    nn: int, optional
      Neighboring system considered from the image can be 6, 18 or 26.
    threshold: float,
      Threshold on labels to define the domain.

    Returns
    -------
    The MultipleROI instance

    Note
    ----
    Only labels > `threshold` are considered

    """
    if isinstance(mim, basestring):
        iim = load(mim)
    else:
        iim = mim

    return mroi_from_array(iim.get_data(), iim.get_affine(), nn, threshold)


def mroi_from_position_and_image(nim, pos):
    """Keep the set of labels of the image corresponding to a certain index
    so that their position is closest to the prescribed one.

    Parameters
    ----------
    mim: NiftiIImage instance, or string path toward such an image
         supposedly a label image
    pos: array of shape(3) or list of length 3,
         the prescribed position

    """
    tmp = mroi_from_image(nim)
    coord = np.array([tmp.domain.coord[tmp.voxels_to_rois_map == k].mean(0)
                      for k in range(tmp.k)])
    idx = ((coord - pos) ** 2).sum(1).argmin()
    return mroi_from_array(nim.get_data() == idx, nim.get_affine())


def mroi_from_balls(domain, positions, radii):
    """Create discrete ROIs as a set of balls within a certain
    coordinate systems.

    Parameters
    ----------
    domain: StructuredDomain instance,
        the description of a discrete domain
    positions: array of shape(k, dim):
        the positions of the balls
    radii: array of shape(k):
        the sphere radii

    """
    # checks
    if np.size(positions) == positions.shape[0]:
        positions = np.reshape(positions, (positions.size), 1)
    if positions.shape[1] != domain.em_dim:
        raise ValueError('incompatible dimensions for domain and positions')
    if positions.shape[0] != np.size(radii):
        raise ValueError('incompatible positions and radii provided')

    label = - np.ones(domain.size)

    for k in range(radii.size):
        supp = np.sum((domain.coord - positions[k]) ** 2, 1) < radii[k] ** 2
        label[supp] = k

    return MultipleROI(domain, label)
