import pysal
import os.path
import gwt
from pysal.weights import W
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["ArcGISTextIO"]

class ArcGISTextIO(gwt.GwtIO):
    """
    Opens, reads, and writes weights file objects in ArcGIS ASCII text format.
    
    Spatial weights objects in the ArcGIS text format are used in 
    ArcGIS Spatial Statistics tools.
    This format is a simple text file with ASCII encoding.
    ArcGIS Spatial Statistics does not specify which file extension to use. 
    Thus, PySAL assumes no extension is used in input ArcGIS text weights files.

    The first line of the ArcGIS text file  is a header including the name of 
    a data column that holded the ID variable in the original source data table.
    After this header line, it includes three data columns 
    for origin id, destination id, and weight values.

    An exemplary structure of an ArcGIS text file is as follows:
    [Line 1]    StationID    
    [Line 2]    1    1    0.0
    [Line 3]    1    2    0.1
    [Line 4]    1    3    0.14286
    [Line 5]    2    1    0.1
    [Line 6]    2    3    0.05
    [Line 7]    3    1    0.16667
    [Line 8]    3    2    0.06667
    [Line 9]    3    3    0.0 
    ...

    As shown in the above example, this file format allows explicit specification 
    of weights for self-neighbors. When no entry is available for self-neighbors, 
    ArcGIS spatial statistics tools consider they have zero weights.
    PySAL ArcGISTextIO class assumes the same.

    References
    ----------
    http://webhelp.esri.com/arcgisdesktop/9.3/index.cfm?TopicName=Modeling_spatial_relationships
    
    Notes
    -----
    When there are an dbf file whose name is identical to the name of the source text file,
    ArcGISTextIO checks the data type of the ID data column and uses it for reading and 
    writing the text file. Otherwise, it considers IDs are strings. 

    """

    FORMATS = ['arcgis_text']
    MODES = ['r', 'w']

    def __init__(self, *args, **kwargs):
        args = args[:2]
        gwt.GwtIO.__init__(self, *args, **kwargs) 

    def _read(self):
        """Reads ArcGIS Text file
        Returns a pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open a text file and read it into a pysal weights object

        >>> w = pysal.open('../../examples/arcgis_txt','r','arcgis_text').read()

        Get the number of observations from the header

        >>> w.n
        3

        Get the mean number of neighbors

        >>> w.mean_neighbors
        3.0

        Get neighbor distances for a single observation

        >>> w['1']
        {'1': 0.0, '3': 0.14285999999999999, '2': 0.10000000000000001}

        """
        if self.pos > 0:
            raise StopIteration

        id_var = self.file.readline().strip()
        self.varName = id_var
        id_order = None
        id_type = str
        try:
            dbf = os.path.join(self.dataPath + '.dbf')
            if os.path.exists(dbf):
                db = pysal.open(dbf, 'r')
                if id_var in db.header:
                    id_order = db.by_col(id_var)
                    id_type = type(id_order[0])
                else:
                    warn("ID_VAR:'%s' was in in the DBF header, proceeding with unordered string ids."%(id_var), RuntimeWarning)
            else:
                warn("DBF relating to ArcGIS TEXT was not found, proceeding with unordered string ids.", RuntimeWarning)
        except:
            warn("Exception occurred will reading DBF, proceeding with unordered string ids.", RuntimeWarning)

        if id_order:
            self.n = len(id_order)
            self.shp = os.path.split(self.dataPath)[1].split('.')[0]
        self.id_var = id_var
        
        weights, neighbors = self._readlines(id_type)
        for k in neighbors:
            if k not in neighbors[k]:
                neighbors[k].append(k)
                weights[k].append(0.0)

        self.pos += 1
        return W(neighbors,weights)

    def write(self, obj):
        """ 

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        an ArcGIS text file
        write a weights object to the opened text file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open('../../examples/arcgis_txt','r','arcgis_text')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w','arcgis_text')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created text file

        >>> wnew =  pysal.open(fname,'r','arcgis_text').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)
        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj),W):
            header = '%s\n' % self.varName
            self.file.write(header)
            self._writelines(obj)
        else:
            raise TypeError, "Expected a pysal weights object, got: %s" % (type(obj))

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)