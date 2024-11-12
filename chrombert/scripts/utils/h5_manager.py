import h5py
import numpy as np

class HDF5Manager:
    def __init__(self, o_file, chunks = True, **kwargs):
        '''
        Initializes an HDF5 file with specified datasets.

        Parameters:
        - o_file (str): The name of the output HDF5 file.
        - chunks: If True, datasets will be chunked automatically. Else, it should be a positive integer (>=2), indicating the chunk size of first dimension. Or use none to disable chunking.
        - kwargs (dict): Keyword arguments where each key is the dataset name and each value is the shape and dtype((*shapes), dtype) of the dataset.
        '''
        self.o_file = o_file
        self.chunks = chunks
        self.kwargs = kwargs
        self.n_samples = 0
        self.file = None

    def __enter__(self):
        # Open the HDF5 file in write mode
        self.file = h5py.File(self.o_file, 'w')
        # Create datasets based on provided shapes
        for key, info in self.kwargs.items():
            shape, dtype = info[0], info[1]
            if self.chunks is True:
                self.file.create_dataset(key, shape=shape, dtype = dtype, chunks = True, maxshape = (None, *shape[1:]))
            elif self.chunks is None:
                self.file.create_dataset(key, shape=shape, dtype = dtype)
            else:
                self.file.create_dataset(key, shape=shape, dtype = dtype, chunks = (self.chunks, *shape[1:]))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the HDF5 file
        if self.file:
            self.file.close()

    def insert(self, **data):
        '''
        Inserts data into the HDF5 file.

        Parameters:
        - data (dict): Keyword arguments where each key is the dataset name and each value is the data to be inserted.
        
        Raises:
        - AssertionError: If keys in data do not match the datasets in the file.
        - AssertionError: If the first dimension of all values is not the same.
        - AssertionError: If inserting more samples than the dataset can hold.
        ''' 
        assert self.file is not None, "File is not open."
        
        # Ensure all keys in data match the datasets in the file
        assert all(key in self.file for key in data.keys()), \
            "All keys in data must match the datasets in the file."
        
        # Ensure the first dimension of all values is the same
        samples = [data[key].shape[0] for key in data.keys()]
        assert len(set(samples)) == 1, "First dimension of all values should be the same."
        
        # Check if the total number of samples will exceed dataset size
        new_samples = samples[0]
        assert self.n_samples + new_samples <= self.file[list(data.keys())[0]].shape[0], \
            "Inserting more samples than the dataset can hold."
        
        # Insert data into the datasets
        for key in data.keys():
            self.file[key][self.n_samples:self.n_samples + new_samples] = data[key]
        
        # Update sample counter
        self.n_samples += new_samples

# Example Usage
if __name__ == "__main__":
    shapes = {
        'dataset1': (1000, 10),
        'dataset2': (1000, 20)
    }

    data1 = np.random.rand(100, 10)
    data2 = np.random.rand(100, 20)

    with HDF5Manager('output.h5', **shapes) as manager:
        manager.insert(dataset1=data1, dataset2=data2)
        # You can call manager.insert multiple times as needed
