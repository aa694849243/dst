from __future__ import print_function
import os
import sys
import re
import glob
from collections import OrderedDict
import json
import h5py

import numpy as np

###### Document Description
Description = """ Utility functions for hdf5 files. """

###### Version and Date
PROG_VERSION = '1.0'
PROG_DATE = '2020-08-03'

###### Usage
usage = """
    Info:    {}
    Version: {}  
    Date:    {}
    Author:  Chen Bichao

    Usage:   Simple usage see test() function. Detail info see docstring of each function. 

    Functions: 
    unpack_data          To unpack data and attributes from hdf5(fast5) file.
    write_to_hdf5        Write data and attribute (in dict) to hdf5 file.

    convert_dtype        H5py datatype conversion, can be used separately to convert datatype to python3 native.
    to_json              Dump dict/h5py file to json.     
    S2U                  Convert numpy structured array's field from ASCII to Unicode. 
    U2S                  Convert numpy structured array's field from Unicode to ASCII. 

    stack_dict_to_path   Reformat nested dict to path-like dict. See docstring of unpack_data.
    path_dict_to_stack   Reformat path-like dict to nested dict. See docstring of unpack_data.

    search_key_in_dict   A generator, input a substring of KEY, Fing full path of the KEY in dict.
    find_val_in_dict     A generator, recursively yield all values with specific key in dict. 

""".format(Description, PROG_VERSION, PROG_DATE)

# set constant values
FILE_EXT = ['.fast5', '.hdf', '.h4', '.hdf4', '.he2', '.h5', '.hdf5', '.he5']

NPY_INT = OrderedDict([(np.int16, np.iinfo(np.int16).max),
                       (np.int32, np.iinfo(np.int32).max),
                       (np.int64, np.iinfo(np.int64).max)])

NPY_FLT = OrderedDict([(np.float16, np.finfo(np.float16).max),
                       (np.float32, np.finfo(np.float32).max),
                       (np.float64, np.finfo(np.float64).max)])


def natural_sort(l):
    """
    Sort string in natural numerical order.

    Input:
        l: list to be sorted.

    Return:
        sorted list.

    Example:
    sorted(): ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    natural_sort(): ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']

    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def open_file(fn, mode='a'):
    """
    Open hdf5 file.

    Inputs:
        fn: path to hdf5 file, for compatibility, add support for h5py.File and h5py.Group
        mode: mode to open hdf5 file, default to 'a'. For more infos: http://docs.h5py.org/en/stable/high/file.html

    Return:
        h5py.File or h5py.Group object.

    """
    # if input is a h5py.File or h5py.Group, return input
    if isinstance(fn, h5py.Group):
        return fn

    try:
        fh = h5py.File(fn, mode)
    except Exception as e:
        print(e)
        print("Error: 1st arg should be either a hdf5 file handle or path to hdf5 file!")
        sys.exit(-1)
    return fh


def to_json(input, output='result.json', save_attr=True, save_data=False):
    """
    Dump input to json.

    Inputs:
        input: can be a dict, list of dicts, file handle or str of hdf5 filename.
        output: filename or file path of the output json file.
        save_attr: boolean, whether to save hdf5 attributes, default to True.
        save_data: boolean, whether to save hdf5 data,  default to False.

        NOTE: 'save_attr' and 'save_data' only take effect when 'input'
        is file handle or str of hdf5 filename.

    Return:
        None.
    """

    def _dtype_check_and_convert(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _dtype_check_and_convert(v)
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()

    # A reminder.
    print('\n', '*' * 50, "\n WARNING: saving numpy structured array to json will loose field name!\n", '*' * 50)

    jf = open(output, 'w')

    if isinstance(input, dict):
        # convert path-like dict to nested dict
        d = path_dict_to_stack(input)

    elif isinstance(input, list):
        # convert all dicts in the list
        d = input
        for i, item in enumerate(d):
            d[i] = path_dict_to_stack(item)

    else:
        # if 'input' is not a dict or list of dicts
        try:
            fh = h5py.File(input, 'r')
        except TypeError:
            # if 'input' is already a file handle
            fh = input
        try:
            data, attr = unpack_data(fh, pretty=True)
        except Exception as e:
            # if 'input' is neither a hdf5 file handle nor a hdf5 file path
            print(e)
            sys.exit(-1)

        # set dump info based on save flags.
        if save_attr and not save_data:
            d = attr
        elif save_attr and save_data:
            d = [attr, data]
        elif not save_attr and save_data:
            d = data
        else:
            print("WARNING: both 'save_attr' and 'save_data' set to False, no info is saved to json.")
            d = []
    try:
        _dtype_check_and_convert(d)
    except:
        for di in d:
            _dtype_check_and_convert(di)

    # dump to json file, with 4 spaces indentation.
    jf.write(json.dumps(d, indent=4))
    jf.close()


def unpack_data(fh, pretty=True, mode='all'):
    """
    Entry function for unpacking data in hdf5 files.

    Inputs:
        fh: file handle of the opened hdf5 file, or full path of the hdf5 file.
        pretty: boolean, whether to reformat the unpacked structure. default to True.
        mode: mode to unpack data, one of 'data', 'attr', 'all', 'data' will unpack only
            data, 'attr' will unpack only attributes, default to 'all'.

    Returns:
        data: python dict{}. Unpacked hdf5 data.
        attr: python dict{}. Unpacked hdf5 attributes.

    Unpacked dict example:
    pretty = False,
    attr = {'/read_a65695bc-0423-40d4-9fcb-fc21317e4964/tracking_id/asic_temp': '32.188072',
            '/read_a65695bc-0423-40d4-9fcb-fc21317e4964/tracking_id/asic_version': 'IA02D',
            ...
           }
    pretty = True,
    attr = {'read_a65695bc-0423-40d4-9fcb-fc21317e4964': {
                'tracking_id': {
                    'asic_temp': '32.188072',
                    'asic_version': 'IA02D'} }
            ...
           }
    """

    if isinstance(fh, str):
        try:
            fh = h5py.File(fh, 'r')
        except:
            print("Error: 1st arg should be either a hdf5 file handle or path to hdf5 file!")
            sys.exit(-1)
    try:
        data, attr = _recursive_unpack(fh, mode)
    except Exception as e:
        print(e)
        sys.exit(-1)

    if pretty:
        attr = path_dict_to_stack(attr)
        data = path_dict_to_stack(data)

    return data, attr


def convert_dtype(val):
    """
    Helper function to clean the data dtype from h5py.Dataset for json dump.
    Alternative solution: _clean() function in ont_fast5_api/data_sanitisation.py.
    """
    if isinstance(val, (np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        val = int(val)
    elif isinstance(val, (np.float64, np.float32, np.float16)):
        val = float(val)
    elif isinstance(val, (np.bool, np.bool_)):
        val = bool(val)
    elif isinstance(val, np.ndarray):
        # check if is structured array (number of dtype > 1)
        if len(val.dtype) > 1:
            # convert any field in val from ascii to unicode
            val = S2U(val)
        else:
            # convert a normal array to list
            val = val.tolist()
            # py3 reads bytes, py2 read string, bytes is not serializable to json
            if isinstance(val[0], bytes):
                val = b''.join(val).decode('utf-8')
    elif isinstance(val, bytes):
        # py3 reads bytes, py2 read string, bytes is not serializable to json
        val = val.decode('utf-8')
    elif isinstance(val, h5py.Empty):
        # Convert h5py.Empty to None. (Empty(dtype=dtype('<f4')) is not JSON serializable)
        val = None
    else:
        # not a numpy dtype/bytes/h5py.Empty
        pass
    return val


def S2U(arr):
    """
    Convert numpy structured array's field from ascii to unicode.
    NOTE: json dump does not support bytes, python3 str default to unicode.
    """
    # loop through each field in the structured array,
    # and convert ascii encoding to unicode encoding
    new_type = []
    for i, dt in enumerate(arr.dtype.descr):
        # set '|S%d' to '<U%d', where %d is determined by old type's itemsize.
        # (e.g. |S16 -> <U16, |S6 -> <U6, etc)
        if arr.dtype[i].type is np.bytes_:
            new_type.append((dt[0], np.dtype(('U', arr.dtype[i].itemsize))))
        else:
            # keep other dtype unchanged
            new_type.append(dt)
    if arr.dtype.descr == new_type:
        return arr
    else:
        return arr.astype(new_type)


def U2S(arr):
    """
    Convert numpy structured array's field from unicode to ascii.
    NOTE: hdf5 best practice: Use numpy.string_ for scalar attributes,
    Use the NumPy S dtype for datasets and array attributes.
    Otherwise, raise TypeError: No conversion path for dtype: dtype('<U16')
    """
    # loop through each field in the structured array,
    # and convert ascii encoding to unicode encoding
    new_type = []
    for i, dt in enumerate(arr.dtype.descr):
        # set '|S%d' to '<U%d', where %d is determined by old type's itemsize.
        # (e.g. |S16 -> <U16, |S6 -> <U6, etc)
        if arr.dtype[i].type is np.str_:
            new_type.append((dt[0], np.dtype(('S', arr.dtype[i].itemsize))))
        else:
            # keep other dtype unchanged
            new_type.append(dt)
    if arr.dtype.descr == new_type:
        return arr
    else:
        return arr.astype(new_type)


def _recursive_unpack(hdf_group, mode='all'):
    """
    Recursively unpack the hdf group.

    Inputs:
        hdf_group: a h5py.Group or h5py.File object.
        mode: mode to unpack data, one of 'data', 'attr', 'all', 'data' will unpack only
            data, 'attr' will unpack only attributes, default to 'all'.
    Returns:
        data: python dict{}, unpacked hdf5 data.
        attr: python dict{}, attributes of the hdf5 file. Ready for json dump.
    """

    data = OrderedDict()
    attr = OrderedDict()

    def _get_attr(hdf_obj, attr):
        # Note that numpy dtype is not serializable to json, thus convert first.
        for k, v in zip(hdf_obj.attrs.keys(), hdf_obj.attrs.values()):
            if hdf_obj.name == '/':
                # top level attributes
                attr['/' + k] = convert_dtype(v)
            else:
                attr[hdf_obj.name + '/' + k] = convert_dtype(v)
        return attr

    # get all attributes directly under current group
    if mode != 'data':
        attr = _get_attr(hdf_group, attr)

    for k in natural_sort(hdf_group.keys()):
        try:
            if isinstance(hdf_group[k], h5py.Dataset):
                # H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
                # data[hdf_group.name+'/'+k] = hdf_group[k].value
                if mode != 'attr':
                    if hdf_group.name == '/':
                        # top level dataset
                        data['/' + k] = convert_dtype(hdf_group[k][()])
                    else:
                        data[hdf_group.name + '/' + k] = convert_dtype(hdf_group[k][()])
                # get attributes from Dataset if there are any
                if mode != 'data':
                    attr = _get_attr(hdf_group[k], attr)
            elif isinstance(hdf_group[k], h5py.Group):
                # recursively unpack
                sub_data, sub_attr = _recursive_unpack(hdf_group[k], mode)
                # transfer sub_data and sub_attr to data and attr
                data.update(sub_data)
                attr.update(sub_attr)
                del sub_data, sub_attr
        except Exception as e:
            # try-except: special treatment for ONT R7 fast5: entry 'InputEvents' has no value.
            continue

    return data, attr


def stack_dict_to_path(d, parent_key=''):
    """
    Reformat stack dict to path dict.

    Inputs:
        d: stack(nested) dict to be reformat.
        parent_key: parent key to be added as prefix for all keys in d.

    Return:
        path_d: reformatted path-like dict.
    """

    def _convert_dict(d, parent_key=''):
        """
        Recursively build the path-like dict key from stack(nested) dict.

        Inputs:
            d: stack(nested) dict to be reformatted.
            parent_key: path-like prefix of current nested level. Default is '', the top level.
        Return:
            out_d: output reformatted path-like dict. Levels of keys are separated by '/'.
        """

        out_d = {}
        for k, v in d.items():
            # convert nested keys to path-like key
            if k.startswith('/'):
                k = parent_key + k
            else:
                k = parent_key + '/' + k
            # Recursively convert nested dicts
            if hasattr(v, 'items'):
                out_d.update(_convert_dict(v, k))
            else:
                out_d[k] = v
        return out_d

    path_d = _convert_dict(d, parent_key)
    return path_d


def path_dict_to_stack(d):
    """
    Reformat path dict to stack dict.

    Input:
        d: path-like dict. Nested keys are separated by '/'. All keys are at the same level.

    Return:
        stack_d: reformatted stack(nested) dict.
    """

    stack_d = {}

    def _set_dict(d, key, dval):
        """
        Recursively search the right level of nested keys and set the value to the dictionary.

        Inputs:
            d: dict{}. Inplace update.
            key: list[]. List of nested keys to iterate. key[-1] is the target place.
            dval: Value to be stored in d['key[0]/key[1]/.../key[-1]'].
        Return:
            No return. Inplace update of dictionary d.
        """

        if len(key) > 1:
            if key[0] not in d.keys():
                d[key[0]] = {}
            _set_dict(d[key[0]], key[1:], dval)
        else:
            d[key[0]] = dval

    for k, v in d.items():
        # split nested keys and remove all empty namespace
        name_list = [n for n in k.split('/') if len(n) > 0]
        if len(name_list) > 0:
            # len(name_list) > 0 indicates there exists nested keys
            _set_dict(stack_d, name_list, v)
        else:
            # Usually this means the root/top of the hdf5 file
            stack_d[k] = v

    return stack_d


def find_val_in_dict(key, var):
    """
    Recursively search value with specific key in dict/or hdf5.Group.

    Inputs:
        key: key to be searched.
        var: dict, nested dict, list of dicts, or hdf5.Group.

    Yield:
        value associated with key if found in var. All values will be yielded if multiple
        entries with the same key are found.
    """

    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in find_val_in_dict(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in find_val_in_dict(key, d):
                        yield result


def search_key_in_dict(key, var):
    """
    Fing KEY in dict, where 'key' is a substring of KEY.

    Input:
        key: str, substring of the needed key.
        var: dict, nested dict, list of dicts.

    Yield:
        the whole structure from root of var to KEY, separated by '/'.
        Example:
        input: ( 'EventDetection', {'Analyses': {'EventDetection_000': {...} } } )
        yield: 'Analyses/EventDetection_000'
    """

    if hasattr(var, 'items'):
        for k, v in var.items():
            if key in k:
                yield k
            if isinstance(v, dict):
                for result in search_key_in_dict(key, v):
                    yield k + '/' + result
            elif isinstance(v, list):
                for d in v:
                    for result in search_key_in_dict(key, d):
                        yield k + '/' + result


def write_to_hdf5(fh, data={}, attr={}):
    """
    Write data and attribute to hdf5 file.

    Inputs:
        fh: hdf5 file handle or str to indicate file name.
        data: nested dict or path-like dict of data to be saved in hdf5.
        attr: nested dict or path-like dict of attributes to be saved in hdf5.

    Return:
        None.
    """

    if isinstance(fh, str):
        try:
            fh = h5py.File(fh, 'a')
        except Exception as e:
            print(e)
            sys.exit(-1)

    """ Recursively create groups and write attributes from dict to hdf5. """

    # inner helper function for writing attribute
    def _write_attrs(gp, d):
        for k, v in d.items():
            if isinstance(v, dict):
                try:
                    # write attr to group
                    sub_gp = gp.require_group(k)
                    _write_attrs(sub_gp, v)
                except TypeError as e:
                    # print(e)
                    # write attr to dataset
                    _write_attrs(gp[k], v)
            else:
                # Temporary fix: convert python None to h5py.Empty with np.float32 as dtype.
                if v is None:
                    v = h5py.Empty(np.dtype(np.float32))
                if isinstance(v, str):
                    if sys.version_info < (3, 0):
                        v = bytes(v)
                    else:
                        v = bytes(v, 'utf8')
                gp.attrs[k] = v

    """ Write data from dict to hdf5. Perform necessary datatype conversion, and handle data overwriting. """

    # inner helper function for writing data
    def _write_data(fh, data):
        for k, v in data.items():
            if isinstance(v, np.ndarray) and len(v.dtype) > 1:
                # convert unicode field in numpy structured array to ascii
                v = U2S(v)
            elif isinstance(v, str):
                # convert string to bytes, then to fix-length numpy 'S' type
                v = np.frombuffer(v.encode('ascii'), dtype=np.dtype(('S', len(v))))
            elif isinstance(v, list):
                # normally list data is an array of numbers
                # python int is default to int64(for 64bit system),
                # python float is default to double-precision
                v = np.array(v)
                # check max of array, find smallest dtype to save to hdf5
                max_v = np.max(v)
                # check dtype.name instead of dtype.type, because numpy has multiple mixed types for int & float
                # more info: https://docs.scipy.org/doc/numpy/user/basics.types.html
                if 'int' in v.dtype.name:
                    for dtype, type_max in NPY_INT.items():
                        if max_v < type_max:
                            v = v.astype(dtype)
                            break
                elif 'float' in v.dtype.name:
                    for dtype, type_max in NPY_FLT.items():
                        if max_v < type_max:
                            v = v.astype(dtype)
                            break
            try:
                # normal dataset creation
                fh.create_dataset(k, data=v)
            except RuntimeError as e:
                # if dataset already exists, check shape, replace if shape matches,
                # else delete before creating new dataset
                print(e, ': ', k)
                if fh[k].shape == v.shape:
                    print("REP: size(new_data) {} == size(old_data) {}, replacing dataset...\n".format(v.shape,
                                                                                                       fh[k].shape))
                    data = fh[k]
                    data[...] = v
                else:
                    print("DEL: size(new_data) {} != size(old_data) {}, delete old and create new dataset...\n".format(
                        v.shape, fh[k].shape))
                    del fh[k]
                    fh.create_dataset(k, data=v)

    # NOTE: data must write before attribute, as some attributes can be saved to h5py.Dataset,
    # which will raise RuntimeError: Unable to create link (name already exists),
    # when saving attribute prior to data.

    # For data, convert nested dict to path-like dict
    data = stack_dict_to_path(data)
    # save data to hdf5
    _write_data(fh, data)

    # For attribute, convert path-like dict to nested dict
    attr = path_dict_to_stack(attr)
    # write attribute to hdf5
    _write_attrs(fh, attr)


def main():
    """ Sample code to use hdf5 utility functions. """

    R7_fast5 = '/mnt/seqdata/Public_shared/Data_analysis_collection/ONT/R73_MAP006/ERR1147230_pass/nanopore2_Ecoli_K12_MG1655_PCR_20150928_1459_1_ch447_file65_strand.fast5'
    # R10_fast5 = '/mnt/seqdata/Public_shared/Data_analysis_collection/ONT/agriculture_institute/mubieguoR10/pass/FAL29659_e8a6867e253692aca252b98082a101dab394503d_2.fast5'
    sample = R7_fast5

    print("Start Simle Testing...")

    data, attr = unpack_data(sample, pretty=True)
    to_json(data, 'test_json.json')
    to_json([data, attr], 'test_json.json')
    to_json(sample, 'test_json.json')
    to_json(h5py.File(sample, 'r'), 'test_json.json', save_data=True)
    print("Test: dump to json completed.\n")

    # attr: double function call to insure consistency
    a1 = path_dict_to_stack(attr)
    stack_a = path_dict_to_stack(a1)
    # attr: convert back to compare with original
    a2 = stack_dict_to_path(stack_a)
    path_a = stack_dict_to_path(a2)
    # data: double function call to insure consistency
    d1 = path_dict_to_stack(data)
    stack_d = path_dict_to_stack(d1)
    # data: convert back to compare with original
    d2 = stack_dict_to_path(stack_d)
    path_d = stack_dict_to_path(d2)
    if a1 != stack_a or a2 != path_a or path_a != attr or \
            d1 != stack_d or d2 != path_d or path_d != data:
        print("Error")
        sys.exit(-1)
    print("Test: stack(nested) dict and path-like dict mutual conversion check completed.\n")

    # write data before attribute, as some attributes in the current sample data store in h5py.Dataset.
    write_to_hdf5('test.h5', data=data)
    # attribute and data can be saved separately
    write_to_hdf5('test.h5', attr=attr)
    # updating and overwriting is possible
    write_to_hdf5('test.h5', data=data, attr=attr)
    # check if output h5 is the same as input fast5
    td, ta = unpack_data('test.h5', pretty=False)
    if attr != ta:
        print("Error: saved h5 is not the same as input!")
        sys.exit(-1)
    for k, v in data.items():
        try:
            if td[k] != v:
                print("Error: saved h5 is not the same as input!")
                sys.exit(-1)
        except:
            if not np.array_equal(td[k], v):
                print("Error: saved h5 is not the same as input!")
                sys.exit(-1)
    print("Test: write to hdf5 check completed.\n")

    # use <nested dict> will produce expected results
    data, attr = unpack_data(sample, pretty=True)
    # get value of 'start_time' from attr
    st = next(find_val_in_dict('start_time', attr))
    # get value of 'Events' from data
    events = next(find_val_in_dict('Events', data))

    # find key name containing 'EventDetection' in attr
    path_attr = path_like_to_stack(data)
    ed = next(search_key_in_dict('EventDetection', data))
    '/Analyses/EventDetection_000'

    st = next(find_val_in_dict(ed, attr))

    # find key name containing 'Fastq' in data
    fq = next(search_key_in_dict('Fastq', data))
    # print(st, events, ed, fq)
    print("Test: find value in dict/search key in dict check completed.\n")


if __name__ == "__main__":
    print(usage)

    # main()

    R7_fast5 = '/mnt/seqdata/output_data/20220324220245_LAB256V2_5K_PC28_30_Z0_HD25j4j12d_AD1_Ecoli_Wangpeiru_Mux/results/Meta/channel17.fast5'
    data, attr = unpack_data(R7_fast5, mode='all')
    data=data['Raw']['Reads']
    pass
    # print(data.keys())
    # print(attr.keys())


