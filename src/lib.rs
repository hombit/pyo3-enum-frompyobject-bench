use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet, PyTuple};

type Arr<'a, T> = PyReadonlyArray1<'a, T>;

#[derive(FromPyObject)]
pub enum GenericFloatArray1<'a> {
    #[pyo3(transparent, annotation = "np.ndarray[float32]")]
    Float32(Arr<'a, f32>),
    #[pyo3(transparent, annotation = "np.ndarray[float64]")]
    Float64(Arr<'a, f64>),
}

#[derive(FromPyObject)]
pub enum Collection<'a> {
    #[pyo3(transparent, annotation = "list")]
    List(&'a PyList),
    #[pyo3(transparent, annotation = "tuple")]
    Tuple(&'a PyTuple),
    #[pyo3(transparent, annotation = "set")]
    Set(&'a PySet),
}
